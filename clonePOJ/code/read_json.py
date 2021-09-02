import json
import os

scores_0 = json.load(open("scores0.json"))
scores_1 = json.load(open("scores1.json"))
scores_2 = json.load(open("scores2.json"))

scores0= {}
scores1 = {}
scores2 ={}
for i  in scores_0.keys():
    scores0[int(i)] = scores_0[i]

for i  in scores_1.keys():
    scores1[int(i)] = scores_1[i]

for i  in scores_2.keys():
    scores2[int(i)] = scores_2[i]

scores = {}
cont = 0
for i in sorted(scores0):
    if cont == 0:
        print(i==38970)
    scores[cont] = scores0[i]
    cont += 1

for i in sorted(scores1):
    scores[cont] = scores1[i]
    cont += 1

for i in sorted(scores2):
    if int(i) >= 46625:
        break
    scores[cont] = scores2[i]
    cont += 1

json.dump(scores,open("scores.json","w"))

scorestxt = open("scores.txt","w")
for i in sorted(scores):
    scorestxt.write(str(i)+"     "+str(scores[i])+"\n")

scorestxt.close()