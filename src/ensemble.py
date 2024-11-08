import json
from collections import Counter

# 載入 JSON 檔案
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# 儲存 JSON 檔案
def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# 載入四個檔案
file_A = load_json('../output/only_chinese/pred_retrieve.json')
file_B = load_json('../output/pos_rank_normalize/pred_retrieve.json')
file_C = load_json('../output/BBAI-1.5/pred_retrieve.json')
file_D = load_json('../output/wo_filter/pred_retrieve.json')

# 將檔案資料轉換為字典，方便查找
def build_dict(data):
    return {item['qid']: item['retrieve'] for item in data['answers']}

dict_A = build_dict(file_A)
dict_B = build_dict(file_B)
dict_C = build_dict(file_C)
dict_D = build_dict(file_D)

# 記錄相同 qid 但不同 retrieve 的項目，並更新 A 中的 retrieve
for qid in dict_A:
    if qid in dict_B and dict_A[qid] != dict_B[qid]:
        # 收集四個檔案中對應的 retrieve 值
        retrieves = [dict_A.get(qid), dict_B.get(qid), dict_C.get(qid), dict_D.get(qid)]
        
        # 計算 retrieve 出現的次數
        retrieve_counts = Counter(retrieves)
        
        # 找出出現次數超過三次的 retrieve 值
        common_retrieve = [retrieve for retrieve, count in retrieve_counts.items() if count > 2]
        
        # 更新 A 檔案的 retrieve 值
        if common_retrieve:
            dict_A[qid] = common_retrieve[0]

# 將更新後的 A 檔案內容轉換回原本的結構
updated_A = {'answers': [{'qid': qid, 'retrieve': retrieve} for qid, retrieve in dict_A.items()]}

# 儲存更新後的 A 檔案
save_json(updated_A, '../output/Final_Answer.json')

print("結果儲存在 Final_Answer.json")
