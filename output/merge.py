import argparse
import os
import json
from collections import defaultdict
parser = argparse.ArgumentParser(description='Process some paths and files.')
parser.add_argument('--folder',type=str,required=True)


args = parser.parse_args()  # 解析參數


total_answers={"answers":[]}

for json_file in os.listdir(args.folder):
    with open(os.path.join(args.folder,json_file,"pred_retrieve.json"),'r') as file:
        part_ans=json.load(file)
        
        total_answers["answers"].extend(part_ans["answers"])
        file.close()
sorted_answers={"answers":sorted(total_answers["answers"], key=lambda x: x["qid"])}
with open(os.path.join(args.folder,"pred_retrieve.json"),'w') as save_file:
    json.dump(sorted_answers,save_file,ensure_ascii=False,indent=4)

