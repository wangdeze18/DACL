import jsonlines
import json
import numpy as np
file_path = './java/codebase.jsonl'

data=[]
with open(file_path) as f:
    for line in f:
        line=line.strip()
        js=json.loads(line)
        data.append(js)

url_to_num ={}
num_to_url ={}
num = 0
for i in range(len(data)):
    print("['original_string']\n")
    print(data[i]['original_string'])
    print("[url]\n")
    print(data[i]['url'])
    url_to_num[data[i]['url']] = num
    num_to_url[num] = url_to_num
    with open("./java_document/"+str(num)+".java","w") as f:
        f.write("class foo{\n" + data[i]['original_string'] + "\n}")

    num+=1

np.save("url_to_num.npy",url_to_num)
np.save("num_to_url.npy",num_to_url)