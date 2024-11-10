import argparse
import os
import json
from collections import defaultdict
parser = argparse.ArgumentParser(description='Process some paths and files.')
parser.add_argument('--folder_a',type=str,required=True)
parser.add_argument('--folder_b',type=str,required=True)
args = parser.parse_args()  # 解析參數

gts= defaultdict(list)
with open("../dataset/preliminary/ground_truths_example.json",'r') as file:
    gts= json.load(file)["ground_truths"]

pred_only_ch=defaultdict(list)
pred_multi=defaultdict(list)
with open(os.path.join(args.folder_a,"pred_retrieve.json"),'r') as file:
    pred_only_ch=json.load(file)["answers"]

with open(os.path.join(args.folder_b,"pred_retrieve.json"),'r') as file:
    pred_multi=json.load(file)["answers"]

total=len(pred_multi)
same=0

for p_m,p_c in zip(pred_multi,pred_only_ch):
    if(p_m["retrieve"]==p_c["retrieve"]):
        same+=1
    else:
        print(f"qid:{p_m['qid']} multilingual:{p_m['retrieve']} only_chinese:{p_c['retrieve']}")
        
print(f"{same}/{total} Precision: {same/total}")