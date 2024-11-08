import argparse
import os
import json
from collections import defaultdict
parser = argparse.ArgumentParser(description='Process some paths and files.')
parser.add_argument('--folder',type=str,required=True)
args = parser.parse_args()  # 解析參數

gts= defaultdict(list)
with open("../dataset/preliminary/ground_truths_example.json",'r') as file:
    gts= json.load(file)["ground_truths"]

pred_retrieves=defaultdict(list)

with open(os.path.join(args.folder,"pred_retrieve.json"),'r') as file:
    pred_retrieves=json.load(file)["answers"]

total=len(gts)
correct=0
for gt,pred in zip(gts,pred_retrieves):
    if(gt["retrieve"]==pred["retrieve"]):
        correct+=1

print(f"Precision: {correct/total}")