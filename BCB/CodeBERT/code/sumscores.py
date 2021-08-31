import numpy as np
score = np.load("scorest.npy")
score0 = np.load("scores0.npy")
score1 = np.load("scores1.npy")
score2 = np.load("scores2.npy")
score3 = np.load("scores3.npy")
print(score1)
sum = score  + (score0 + score1 + score2)/3 
y_preds = sum > 0.99*2
y_preds = score >0.50
def read_answers(filename):
    answers = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            idx1, idx2, label = line.split()
            answers.append(int(label))
    return answers

y_trues = read_answers("../dataset/test.txt")

from sklearn.metrics import recall_score,precision_score,f1_score

scores={}
scores['Recall']=recall_score(y_trues, y_preds, average='macro')
scores['Prediction']=precision_score(y_trues, y_preds, average='macro')
scores['F1']=f1_score(y_trues, y_preds, average='macro')
print(str(scores))