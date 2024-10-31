import json
import os
import argparse
import torch
import numpy as np
from utils.utils import load_data
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification,Trainer, TrainingArguments
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from tqdm import tqdm
# Function to calculate Precision@1
def precision_at_1(pred):
    # Get the highest-scoring prediction for each query
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    # Precision@1 is simply accuracy here because we care only about the top result
    precision_at_1 = accuracy_score(labels, preds)
    return {"precision@1": precision_at_1}
def preprocess_sample(q_data,gt_data,category_data):
    samples=[]
    labels=[]
    for item in q_data:
        q_id=item["qid"]
        curr_gt=gt_data[q_id-1]["retrieve"]
        source =item["source"] 
        main_question =item["query"]
        cgy =item["category"]
        for s in source:
            lb=1 if s == curr_gt else 0
            if(cgy=="faq"):
                qas=category_data[cgy][s]
                combined_ans = " ".join(
                    [ans for qa_pair in qas for ans in qa_pair["answers"]]
                )
                samples.append((main_question,combined_ans,cgy))
                labels.append(lb)
                ## augumentation question
                # for qa in qas:
                #     question=qa["question"]
                #     ans=" ".join(qa["answers"])
                #     self.samples.append((question,ans,cgy))
                #     self.labels.append(lb)
            else:
                ans=category_data[cgy][str(s)]
                samples.append((main_question,ans,cgy))
                labels.append(lb)
    return samples,labels


class ChunkedTrainer(Trainer):
    def compute_loss(self, model, inputs,num_items_in_batch):
        # Extract input_ids, attention_mask, and labels
        input_ids = inputs['input_ids']  # Shape: (chunk, batch, length)
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']

        # Accumulate losses across chunks
        total_loss = 0
        chunk_count = input_ids.size(0)  # Number of chunks

        for chunk_idx in range(chunk_count):
            # Get inputs for the current chunk
            chunk_input_ids = input_ids[chunk_idx]       # Shape: (batch, length)
            chunk_attention_mask = attention_mask[chunk_idx]  # Shape: (batch, length)
            
            # Forward pass
            outputs = model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask, labels=labels)
            chunk_loss = outputs.loss
            
            # Accumulate loss
            total_loss += chunk_loss

        # Average loss across chunks (optional)
        avg_loss = total_loss / chunk_count
        return avg_loss

class CustomQuestionDataset(Dataset):
    def __init__(self, samples,labels):
        self.max_length=128
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.samples=samples
        self.labels=labels
    def decode_token(self,inputs):
        for id, input_ids in enumerate(inputs["input_ids"]):
            print(id, len(input_ids), self.tokenizer.decode(input_ids))
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        question,answers,cgy=self.samples[idx]

        ## different answer has different length, need truncate to same length
        inputs=self.tokenizer(
            text=question,
            text_pair=answers,
            max_length=self.max_length,
            padding="max_length",
            truncation="only_second",
            return_tensors="pt",           
            stride=10, # move forward 50 (overlap 50 chars)
            return_overflowing_tokens=True,
            return_offsets_mapping=True
            
        )
        #print(inputs["input_ids"].shape,inputs["attention_mask"].shape,self.labels[idx])
        if(inputs["input_ids"].shape[1] != self.max_length):
            self.decode_token(inputs)    
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": self.labels[idx]
        }
# Custom collate function for DataLoader
def custom_collate_fn(batch):
    # Stack tensors and ensure padding to the max length
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    padding_input_ids=pad_sequence(input_ids,batch_first=False,padding_value=0)
    padding_attention_mask=pad_sequence(attention_mask,batch_first=False,padding_value=0)
    labels = torch.tensor([item['labels'] for item in batch])
    print(padding_input_ids.shape,labels.shape)
    return {
        "input_ids": padding_input_ids,
        "attention_mask": padding_attention_mask,
        "labels": labels
    }
if __name__ == "__main__":
    # 使用argparse解析命令列參數
    
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--epoches',type=int,default=100)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--lr',type=float,default=1e-3)
    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    with open(os.path.join(args.question_path,"questions_example.json"), 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案
    with open(os.path.join(args.question_path,"ground_truths_example.json"), 'rb') as f:
        gt_ref = json.load(f)
    source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
    
    corpus_dict_insurance = load_data(source_path_insurance,'insurance.json')

    source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
    corpus_dict_finance = load_data(source_path_finance,"finance.json")
    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
    category_data={
        "insurance":corpus_dict_insurance,
        "finance":corpus_dict_finance,
        "faq":key_to_source_dict
    }
    samples,labels=preprocess_sample(qs_ref["questions"],gt_ref["ground_truths"],category_data)
    dataset=CustomQuestionDataset(samples,labels)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,collate_fn=custom_collate_fn)
    # for batch in dataloader:
    #     #print(batch["input_ids"].shape)
    #     continue
    # exit()
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset,val_dataset,test_dataset=random_split(dataset,[train_size,valid_size,test_size])
    train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=custom_collate_fn)
    val_loader=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=custom_collate_fn)
    test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=custom_collate_fn)
    model_path="../output/model.pth"
    
    
    if(not os.path.exists("../output/model.pth")):
        model = BertForSequenceClassification.from_pretrained("bert-base-chinese")
        training_args=TrainingArguments(
            output_dir='../output',
            num_train_epochs=3,
            per_device_train_batch_size=args.batch_size,
            evaluation_strategy="epoch"
        )
        trainer=ChunkedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=custom_collate_fn,
            compute_metrics=precision_at_1
            
        )
        trainer.train()
        trainer.save_model(model_path)
    else:
        model = BertForSequenceClassification.from_pretrained(model_path)
        
    

    model.eval()
    with torch.no_grad():
        test_preds=[]
        test_labels=[]
        for test_batch in tqdm(dataloader):
            input_ids=test_batch["input_ids"]
            attention_mask=test_batch["attention_mask"]
            labels=test_batch["label"]
            outputs=model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
            preds=outputs.logits.argmax(-1)
            test_preds.append(preds)
            test_labels.append(labels)
        test_preds=torch.cat(test_preds)
        test_labels=torch.cat(test_labels)
        print(test_preds,test_labels)
        precision=accuracy_score(test_labels,test_preds)
        print(f"precision: {precision}")
    #torch.save(model.state_dict(),"../output/model.pth")
    ## testing

    ## self training
    # device=torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    # model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
    # model=model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # for epoch in args.epoches:
    #     model.train()
    #     total_loss=0.0
    #     for batch_idx,batch in enumerate(train_loader):
    #         input_ids=batch["input_ids"].to(device)
    #         attention_mask=batch["attention_mask"].to(device)
    #         outputs=model(input_ids=input_ids,attention_mask=attention_mask,labels=None)
    #         loss=outputs.loss
    #         total_loss += loss.item()
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    # print(f"Epoch {epoch + 1}/{args.epoches } | Loss: {total_loss / len(train_loader)}")
    # # 將答案字典保存為json文件
    # with open(args.output_path, 'w', encoding='utf8') as f:
    #     json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符