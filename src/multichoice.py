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
from utils.qa_retriever import Retriever
from utils.qa_reranker import Reranker
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
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
                ans_len=len(ans)
                pivot=2000
                if(ans_len>pivot):
                    parts=ans_len  // pivot 
                    split_ans=[ans[i * pivot:(i+1) * pivot] for i in range(parts)]
                    split_ans.append(ans[parts * pivot:])
                    for sub_ans in split_ans:
                        samples.append((main_question,sub_ans,cgy))
                        labels.append(lb)
                else:
                    samples.append((main_question,ans,cgy))
                    labels.append(lb)
    return samples,labels

def preprocess_faq(key_to_source_dict):
    full_context={}
    for key,source in key_to_source_dict.items():
        full_context[key]=""
        for s in source:
            full_context[key]+= s["question"]
            for ans in s["answers"]:
                full_context[key]+= ans
    return full_context
def split_chunk(page_content,max_len=256,overlap_len=100):
    cleaned_chunks = []
    i = 0
    # 暴力将整个pdf当做一个字符串，然后按照固定大小的滑动窗口切割
    all_str = ''.join(page_content)
    all_str = all_str.replace('\n', '')
    while i < len(all_str):
        cur_s = all_str[i:i+max_len]
        if len(cur_s) > 10:
            cleaned_chunks.append(cur_s)
        i += (max_len - overlap_len)

    return cleaned_chunks
if __name__ == "__main__":
    # 使用argparse解析命令列參數
    
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--error_path',type=str,default="../output/error_retrieve.json",help="錯誤答案分析")
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
    corpus_dict_insurance={int(id):context for id,context in corpus_dict_insurance.items()}
    corpus_dict_finance={int(id):context for id,context in corpus_dict_finance.items()}

    
    #samples,labels=preprocess_sample(qs_ref["questions"],gt_ref["ground_truths"],category_data)
    faq_corpus=preprocess_faq(key_to_source_dict)
    for id,corpus in faq_corpus.items():
        faq_corpus[id]=split_chunk(corpus)
    for id,corpus in corpus_dict_insurance.items():
        corpus_dict_insurance[id]=split_chunk(corpus)
    for id,corpus in corpus_dict_finance.items():
        corpus_dict_finance[id]=split_chunk(corpus)
    category_corpus={
        "insurance":corpus_dict_insurance,
        "finance":corpus_dict_finance,
        "faq":faq_corpus
    }

    # faq_questions=[item for item in qs_ref["questions"] if item["category"]=="insurance"]
    # gt_faq=[item for item in gt_ref["ground_truths"] if item["category"]=="insurance"]
    correct=0
    qs=qs_ref["questions"]
    gt=gt_ref["ground_truths"]
    error_answer=[]
    truth_answer={"answers":[]}
    for i,item in tqdm(enumerate(qs)):
        q_id=item["qid"]
        gt_res_id= next((item["retrieve"] for item in gt if item["qid"]==q_id),None)
        s_id=item["source"]
        q=item["query"]
        cgy=item["category"]
        
        corpus=category_corpus[cgy]
        
        source_ans={id:corpus[id] for id in corpus.keys() if id in s_id}
        retriever = Retriever(emb_model_name_or_path="BAAI/bge-large-zh-v1.5", corpus=source_ans)
        reranker = Reranker(rerank_model_name_or_path="BAAI/bge-reranker-v2-m3")
        retrieve_ans= retriever.retrieval(q)
        rerank_res_ids,rerank_scores = reranker.rerank(retrieve_ans, q, k=1)
        predicted_id=rerank_res_ids[0]
        
        info={
            "qid":q_id,
            "query": q,
            "category":cgy,
            "retrieve":gt_res_id,
            "predicted":predicted_id,
            "source":str(s_id),
            "rank_ids":str(rerank_res_ids),
            "rank_scores":str(rerank_scores)
        }
        if(predicted_id==gt_res_id):
            correct+=1
            truth_answer["answers"].append(info)
        else:
            error_answer.append(info)

        print(f"Current correct:{correct}/{i+1}")
        print(f"Current accuracy:{correct/(i+1)}")
    with open(args.output_path,'w', encoding='utf8') as f:
        json.dump(truth_answer, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
    with open(args.error_path, 'w', encoding='utf8') as f:
        
        json.dump(error_answer, f, ensure_ascii=False,indent=4)
    print(f"Precision: {correct/len(gt)}")