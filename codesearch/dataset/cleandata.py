import os
import numpy as np
transDic = {}

for filepath, dirnames, filenames in os.walk("java_document/"):
    for filename in filenames:
        source_file = os.path.join(filepath, filename)
        transDic[filename] = []

# remove the foo()
def cleanCla(source_file,filename):

    content = open(source_file,"r")
    lines = content.readlines()
    newl = "".join(lines[1:-1])
    print(newl)
    #filename = filename.replace('n',"")
    fileid = filename[:-5] + "_" + str(len(transDic[filename])) + ".java"
    transDic[filename].append(fileid)
    newfilename = "./smallClean/" + fileid
    print(fileid)
    with open(newfilename,"w") as f:
        f.write(newl)

for filepath, dirnames, filenames in os.walk("transformDir/"):
    for filename in filenames:
        source_file = os.path.join(filepath, filename)
        #keyFile = filename.replace('n', "")
        cleanCla(source_file,filename)

np.save("smallDict.npy",transDic)