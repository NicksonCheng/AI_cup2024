import torch
import math
from collections import defaultdict
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Reranker:
    def __init__(self, rerank_model_name_or_path, pos_rank=False, device='cuda'):
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name_or_path)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name_or_path)\
            .half().to(device).eval()
        self.pos_rank=pos_rank
        self.device = device
        print('successful load rerank model')

    def normalized(self,score):
        min_s=min(score)
        max_s=max(score)

        normalize_score=[ (s-min_s) / (max_s-min_s)  for s in score]
        return normalize_score
    def rerank(self, docs, query, k=5):
        # docs_ = []
        # for item in docs:
        #     if isinstance(item, str):
        #         docs_.append(item)
        #     else:
        #         docs_.append(item.page_content)
        pairs = []
        for d in docs:
            pairs.append([query, d[1]])
        with torch.no_grad():
            inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)\
                .to(self.device)
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().tolist()
        docs = [(docs[i][0], scores[i]) for i in range(len(docs))]
        docs = sorted(docs, key = lambda x: x[1], reverse = True)

        if(not self.pos_rank):
            doc_id=[doc[0] for doc in docs]
            doc_score=[doc[1] for doc in docs]
            return doc_id,doc_score

        ## used position ranking strategy
        rank_id=[doc[0] for doc in docs]
        ## normalize
        rank_score= [doc[1] for doc in docs]
        normalize_score=self.normalized(rank_score)

        ## softmax
        #rank_score=torch.tensor([doc[1] for doc in docs])
        #softmax_score=F.normalize(rank_score)
        pos_weight_score=[score / math.log2(i+2) for i,score in enumerate(normalize_score)]
        
        sum_score=defaultdict(float)
        for id, score in zip(rank_id,pos_weight_score):
            sum_score[id]+=score
        pos_rank_item=sorted(sum_score.items(), key=lambda x:x[1],reverse=True)

        pos_rank_id=[item[0] for item in pos_rank_item]
        pos_rank_score=[item[1] for item in pos_rank_item]
        return pos_rank_id,pos_rank_score