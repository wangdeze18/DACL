###build trans train.txt,valid.txt,test.txt(large)

import numpy as np
import random
import json
import jsonlines
smallDict = np.load("smallDict.npy",allow_pickle=True).item()
url_to_num = np.load("url_to_num.npy",allow_pickle=True).item()
num_to_url = np.load("num_to_url.npy",allow_pickle=True).item()
#print(smallDict)

def transfertrain(origin_data,output_data):
    data = []
    with open(origin_data,"r") as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            data.append(js)
            num = url_to_num[js['url']]
            translist = smallDict[str(num)+".java"]
            for transfile in translist:
                transf = open("./smallClean/"+transfile,"r")
                js['original_string'] = transf.read()
                transf.close()
                data.append(js)

    with jsonlines.open(output_data, mode='w') as writer:
        for i in data:
            writer.write(i)


def transfervalid(origin_data,output_data):
    data = []
    with open(origin_data,"r") as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            data.append(js)
            num = url_to_num[js['url']]
            translist = smallDict[str(num)+".java"]
            for transfile in translist:
                loc = transfile.find("_")
                suffix = transfile[loc:loc+2]
                temp = js
                temp['url'] = js['url'] + suffix
                data.append(temp)

    with jsonlines.open(output_data, mode='w') as writer:
        for i in data:
            writer.write(i)

def select(file_key):
    if len(smallDict[file_key]) ==0:
        return file_key
    return random.sample(smallDict[file_key],1)[0]

def transfer(origin_data, output_data):
    data = []
    with open(origin_data, "r") as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            num = url_to_num[js['url']]

            transfile = select(str(num) + ".java")
            loc = transfile.find("_")
            suffix = transfile[loc:loc + 2]
            temp = js
            temp['url'] = js['url'] + suffix
            data.append(temp)

    with jsonlines.open(output_data, mode='w') as writer:
        for i in data:
            writer.write(i)


#transfertrain('./java/train.jsonl','./java/large_train.jsonl')
transfervalid('./java/valid.jsonl','./java/large_valid.jsonl')
transfer('./java/test.jsonl','./java/test0.jsonl')
transfer('./java/test.jsonl','./java/test1.jsonl')
transfer('./java/test.jsonl','./java/test2.jsonl')
transfer('./java/test.jsonl','./java/test3.jsonl')

