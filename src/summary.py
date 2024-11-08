import torch
import json
import os
import re
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
from opencc import OpenCC

parser= argparse.ArgumentParser(description='Process some paths and files')
parser.add_argument('--task',type=str,required=True,help="summary insurance or finance")
parser.add_argument("--gpu", type=int, default=0, help="gpu devices")
args=parser.parse_args()

cc = OpenCC('s2t')
model_path = "twwch/mt5-base-summary"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()


def _split_text(text, length):
    chunks = []
    start = 0
    while start < len(text):
        if len(text) - start > length:
            pos_forward = start + length
            pos_backward = start + length
            pos = start + length
            while (pos_forward < len(text)) and (pos_backward >= 0) and (pos_forward < 20 + pos) and (
                    pos_backward + 20 > pos) and text[pos_forward] not in {'.', '。', '，', ','} and text[
                pos_backward] not in {'.', '。', '，', ','}:
                pos_forward += 1
                pos_backward -= 1
            if pos_forward - pos >= 20 and pos_backward <= pos - 20:
                pos = start + length
            elif text[pos_backward] in {'.', '。', '，', ','}:
                pos = pos_backward
            else:
                pos = pos_forward
            chunks.append(text[start:pos + 1])
            start = pos + 1
        else:
            chunks.append(text[start:])
            break
    # Combine last chunk with previous one if it's too short
    if len(chunks) > 1 and len(chunks[-1]) < 100:
        chunks[-2] += chunks[-1]
        chunks.pop()
    return chunks


def summary(text):
    chunks = _split_text(text, 300)
    chunks = [
        "summarize: " + chunk
        for chunk in chunks
    ]
    input_ids = tokenizer(chunks, return_tensors="pt",
                          max_length=512,
                          padding=True,
                          truncation=True).input_ids.to(device)
    outputs = model.generate(input_ids, max_length=250, num_beams=4, no_repeat_ngram_size=2)
    tokens = outputs.tolist()
    output_text = [
        tokenizer.decode(tokens[i], skip_special_tokens=True)
        for i in range(len(tokens))
    ]
    output_text=''.join(output_text)
    filtered_text = re.sub(r'[^\w\s]', '', output_text)
    
    # for i in range(len(output_text)):
    #     traditional_text=cc.convert(output_text[i])
    #     print(output_text[i])
    return filtered_text


# text="台灣水泥股份有限公司及子公司合併權益變動表民國111年及110年1月1日至3月31日僅經核閱未依一般公認審計準則查核歸屬於本公司業主之權益股本保留盈餘其他權益項目普通股股本特別股股本債券換股權利證書資本公積法定盈餘公積特別盈餘公積未分配盈餘合計國外營運機構財務報表換算之兌換差額透過其他綜合損益按公允價值衡量之金融資產為實現評價損益避險工具損益庫藏股票本公司業主權益淨額非控制權益權益合計代碼A1110年1月1日盈餘D1淨利D3其他宗和損益D5110年1月1日至3月31日宗和損益總額M5取得或處分子公司股權價格與帳面價值差額I1可轉換公司債轉換M7對子公司所有權權益變動Z1110年3月31日餘額A1111年1月1日餘額D1淨利D3其他綜合損益D5111年1月1日至3月31日綜合損益總額M7對子公司所有權權益變動Z1111年3月31日餘額"
# summary_text=summary(text)
# print(summary_text)
# exit()



corpus_path="../dataset/preliminary"


if args.task == "finance":
    print("------------------finance--------------------")
    finance_corpus_dict=None
    with open(os.path.join(corpus_path,"finance_all_str.json"),"r") as file:
        finance_corpus_dict=json.load(file)
        file.close()
    summary_finance_corpus_dict={}
    for id,context in tqdm(finance_corpus_dict.items()):
        summary_txt=summary(context)
        print(summary_txt)
        summary_finance_corpus_dict[id]=summary_txt

    with open(os.path.join(corpus_path,"finance_summary.json"),'w') as save_file:
        json.dump(summary_finance_corpus_dict,save_file,ensure_ascii=False,indent=4)
        save_file.close()
elif args.task == "insurance":
    ## insurance
    print("------------------insurance--------------------")
    insurance_corpus_dict=None
    with open(os.path.join(corpus_path,"insurance_all_str.json"),"r") as file:
        insurance_corpus_dict=json.load(file)
        file.close()
    summary_insurance_corpus_dict={}


    for id,context in tqdm(insurance_corpus_dict.items()):
        summary_txt=summary(context)
        print(summary_txt)
        summary_insurance_corpus_dict[id]=summary_txt

    with open(os.path.join(corpus_path,"insurance_summary.json"),'w') as save_file:
        json.dump(summary_insurance_corpus_dict,save_file,ensure_ascii=False,indent=4)
        save_file.close()
else:
    print("No Such file to summarize")
