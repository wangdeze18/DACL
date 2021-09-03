import numpy as np
import json
from tqdm import tqdm, trange
import multiprocessing
#pool = multiprocessing.Pool()

score = np.load("scores_ori.npy")
score0 = np.load("scores0.npy")
score1 = np.load("scores1.npy")
score2 = np.load("scores2.npy")
#print(score)


def builddataset(file_path):
    examples = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            examples.append(js['url'])
    return examples


nl_urls = []
query_examples = builddataset('./dataset/java/test.jsonl')
for example in query_examples:
    nl_urls.append(example)


code_urls = []
code_examples = builddataset('./dataset/java/codebase.jsonl')
for example in code_examples:
    code_urls.append(example)

'''
code_urls = np.load("codeurl1.npy",allow_pickle=True)


code_urls_ori = []
code_examples = builddataset('./dataset/java/codebase.jsonl')
for example in code_examples:
    code_urls_ori.append(example)


temp= score.copy()
print(len(score[0]))
print(len(code_urls))
print(len(code_urls_ori))
count =0
for i in range(len(code_urls)):

    for j in range(len(code_urls_ori)):

        if code_urls[i] == code_urls_ori[j]:
            temp[:,i] = score[:,j]
            count +=1
            break
print(count)
score_ori = temp
scores = score_ori
np.save("scores_ori.npy", score_ori)
'''
scores = score + (score0 + score1 + score2)/3


sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
ranks = []
for url, sort_id in zip(nl_urls, sort_ids):
    rank = 0
    find = False
    for idx in sort_id[:1000]:
        if find is False:
            rank += 1
        if code_urls[idx] == url:
            find = True
    if find:
        ranks.append(1 / rank)
        #print(rank)
    else:
        ranks.append(0)
        #print(docstrings[])
        #print()
        #print(0)

result = {
    "eval_mrr": float(np.mean(ranks))
}
print(result)
