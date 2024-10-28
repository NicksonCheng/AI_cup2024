import json
import os
import argparse
import torch
import numpy as np
from utils.utils import load_data
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import random_split

class CustomQuestionDataset(Dataset):
    def __init__(self, q_data,gt_data,category_data,max_length=100):
        self.max_length=max_length
        self.gt_data=gt_data["ground_truths"]
        self.q_data= q_data["questions"]
        self.category_data=category_data
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.samples=[]
        self.preprocess_sample()
    def preprocess_sample(self):

        for item in self.q_data:
            q_id=item["qid"]
            source =item["source"] 
            main_question =item["query"]
            cgy =item["category"]
            for s in source:
                if(cgy=="faq"):
                    qas=self.category_data[cgy][s]
                    combined_ans = " ".join(
                        [ans for qa_pair in qas for ans in qa_pair["answers"]]
                    )
                    self.samples.append((main_question,combined_ans))
                    for qa in qas:
                        question=qa["question"]
                        ans=" ".join(qa["answers"])
                        self.samples.append((question,ans))
                else:
                    ans=self.category_data[cgy][str(s)]
                    self.samples.append((main_question,ans))

    def decode_token(self,inputs):
        for id, input_ids in enumerate(inputs["input_ids"]):
            print(id, len(input_ids), self.tokenizer.decode(input_ids))
    def __len__(self):
        return len(self.q_data)

    def __getitem__(self, idx):
        question,answers=self.samples[idx]
        inputs=self.tokenizer(
            text=question,
            text_pair=answers,
            max_length=self.max_length,
            truncation="only_second",
            stride=50, # move forward 50 (overlap 50 chars)
            return_overflowing_tokens=True,
            #return_tensors="pt"
        )
        #self.decode_token(inputs)

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze()
        }
if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--batch_size',type=int,default=64)
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
    dataset=CustomQuestionDataset(qs_ref,gt_ref,category_data)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    for batch_samples in dataloader:
        print(batch_samples) 
        exit()
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset,val_dataset,test_dataset=random_split(dataset,[train_size,valid_size,test_size])
    train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)
    
    device=torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
    model=model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in args.epoches:
        model.train()
        total_loss=0.0
        for batch_idx,batch in enumerate(train_loader):
            input_ids=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            outputs=model(input_ids=input_ids,attention_mask=attention_mask,labels=None)
            loss=outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    print(f"Epoch {epoch + 1}/{args.epoches } | Loss: {total_loss / len(train_loader)}")
    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符