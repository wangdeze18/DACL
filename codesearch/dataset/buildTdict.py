import os

transDic = {}

for filepath, dirnames, filenames in os.walk("java_document/"):
    for filename in filenames:
        source_file = os.path.join(filepath, filename)
        transDic[filename] = []


for filepath, dirnames, filenames in os.walk("smallClean/"):
    for filename in filenames:
        source_file = os.path.join(filepath, filename)
        if filename.find("_") !=-1:
            loc = filename.find("_")
            filename = filename[:loc] + '.java'
        transDic[filename].append(source_file)

for k in transDic.keys():
    if len(transDic[k]) == 0:
        print(k)

import numpy as np
np.save("Tdict.npy",transDic)