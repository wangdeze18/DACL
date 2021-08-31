import os
import  numpy as np
import json
import jsonlines
smallDict = np.load("smallDict.npy",allow_pickle=True).item()
url_to_num = np.load("url_to_num.npy",allow_pickle=True).item()
num_to_url = np.load("num_to_url.npy",allow_pickle=True).item()

data = []
with open('./java/codebase.jsonl',"r") as f:
    for line in f:
        line = line.strip()
        js = json.loads(line)
        data.append(js)


for filepath,dirnames,filenames in os.walk("smallClean"): #cleanTransdata
    for filename in filenames:
        sourcefile = os.path.join(filepath,filename)
        with open(sourcefile,"r") as f:
            con = f.read()

        loc = filename.find('_')
        index = int(filename[:loc])
        suffix = filename[loc:loc+2]
        print(suffix)

        temp = data[index]
        temp['original_string'] =  con
        temp['url'] = temp['url'] + suffix

        data.append(temp)

with jsonlines.open("./java/large_codebase.jsonl",mode='w') as writer:
    for i in data:
        writer.write(i)
