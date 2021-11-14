import os
import subprocess
import re

def write_file(filename,a_list):
    fw = open(filename, 'w')
    for i in range(len(a_list)):
        fw.write(a_list[i] +'\n')
    fw.close()

def read_file(filename):
    f = open(filename, encoding='unicode_escape')
    doc_str = f.read()
    f.close()
    return doc_str

def correct_define(cpp_name,error_char):
    label = "using namespace std;"
    doc_str = read_file(cpp_name)
    index = doc_str.find(label)
    if index != -1:
        fw = open(cpp_name,'w')
        new_str = doc_str[:index+len(label)] + "\n" + "#define" +" " +error_char +" " + str(1000) + doc_str[index+len(label):]
        fw.write(new_str)
        fw.close()
    else:
        print(cpp_name,"correct_define exception")


#correct_define("13.cpp","MAX")


def correct_gets(cpp_name):
    label = "gets("
    doc_str = read_file(cpp_name)
    index = doc_str.find(label)
    while(index != -1):
        index_r = doc_str.find(')',index)
        content = doc_str[:index_r][index+5:]
        doc_str = doc_str[:index] + "\n" + "cin>>" + content + doc_str[index_r+1:]

        index = doc_str.find(label)

    #print(doc_str)
    fw = open(cpp_name, 'w')
    fw.write(doc_str)
    fw.close()

def correct_int_main(cpp_name):
    label = "main()"
    doc_str = read_file(cpp_name)
    index = doc_str.find(label)
    if index != -1:
        fw = open(cpp_name, 'w')
        new_str = doc_str[:index] + "int " + doc_str[index:]
        #print(new_str)
        #fw.write(new_str)
        #fw.close()
    else:
        fw = open(cpp_name, 'w')
        label ="main ()"
        index = doc_str.find(label)
        new_str = doc_str[:index] + "int main()" + doc_str[index+len(label):]
        #print(new_str)
        fw.write(new_str)
        fw.close()
        #print(cpp_name, "correct_int_main exception")

def correct_none_return(cpp_name):
    label = "return ;"
    doc_str = read_file(cpp_name)
    index = doc_str.find(label)
    if index != -1:
        fw = open(cpp_name, 'w')
        new_str = doc_str[:index] + "return 0;" + doc_str[index+len(label):]
        #print(new_str)
        fw.write(new_str)
        fw.close()
    else:
        print(cpp_name, "correct_none_return exception")

def correct_ambiguous(cpp_name,error_char):
    doc_str = read_file(cpp_name)

    new_str = re.sub(error_char,error_char+"_1",doc_str)
    #print(new_str)

    fw = open(cpp_name, 'w')
    fw.write(new_str)
    fw.close()

def correct_missing(cpp_name,line_num_list):

    f = open(cpp_name)
    lines = f.readlines()
    f.close()
    for i in range(0,len(line_num_list),2):
        num = line_num_list[i] - 1
        #print(num)
        #print(lines[num])
        lines[num] = lines[num].replace("\n"," ")
        #print(lines[num])


    doc_str = "".join(lines)
    fw = open(cpp_name, 'w')
    fw.write(doc_str)
    fw.close()
    #print(doc_str)

def excute_cmd(cmd):
    cmd = cmd.split(" ")
    p = subprocess.run(cmd, stderr=subprocess.PIPE, shell=False, timeout=145)
    err = p.stderr
    err = str(err, encoding='utf-8', errors='ignore')
    return err


error_file = []
error_list = []
cppdir = "/home/CodeXGLUE-main/Code-Code/Clone-detection-POJ-104/dataset/ProgramData"
num = 0
for filepath, dirnames, filenames in os.walk(cppdir):  
    num += 1
    print(num)
    for filename in filenames:
        source_file = os.path.join(filepath, filename)
        cmd = "g++ " + source_file
        err = excute_cmd(cmd)

        if err.find("error: missing terminating \" character") != -1:
            to_find = re.compile(r':(\d+):\d+: error: missing terminating " character')
            find_list = re.findall(to_find, err)
            line_num_list = list(map(int, find_list))
            # print("1", line_num_list)

            correct_missing(source_file, line_num_list)
            err = excute_cmd(cmd)

        elif err.find("error: missing terminating \' character") != -1:
            to_find = re.compile(r":(\d+):\d+: error: missing terminating ' character")
            find_list = re.findall(to_find, err)
            line_num_list = list(map(int, find_list))

            correct_missing(source_file, line_num_list)
            err = excute_cmd(cmd)
        elif err.find("warning: ISO C++ forbids declaration of ‘main’ with no type") != -1:
            # print(source_file)
            correct_int_main(source_file)
            err = excute_cmd(cmd)

        if err.find("error") != -1:
            error_file.append(source_file)
            error_list.append(err)
        '''
        
        elif err.find("error: ‘gets’ was not declared in this scope;") != -1:
            #print("error 0: gets\n")
            correct_gets(source_file)
        elif err.find("error: use of undeclared identifier") != -1:
            #print("error 1: undeclared identifier\n")
            index = err.find("error: use of undeclared identifier")
            # print(err)
            index_begin = err.find('\'', index)
            index_end = err.find('\'', index_begin + 1)
            to_define_char = err[:index_end][index_begin + 1:]

            correct_define(source_file, to_define_char)
        elif err.find(" was not declared in this scope") != -1: #error: ‘MAX’
            #print("error 2: not declared scope\n")
            index = err.find(" was not declared in this scope")
            err_part = err[:index-1]

            index_begin = err_part.rfind('‘')
            #print(err_part)
            index_end = index - 1
            #print(index_begin,index_end)
            to_define_char = err[:index_end][index_begin + 1:]
            #print(to_define_char)
            correct_define(source_file, to_define_char)
        elif err.find("error: C++ requires a type specifier for all declarations") != -1:
            #print("error 3: none main()\n")
            correct_int_main(source_file)
        elif err.find("error: return-statement with no value, in function returning ‘int’")!=-1:
            #print("error 4: none return\n")
            correct_none_return(source_file)
        elif err.find("error: reference to ")!=-1: # re.search(r"error: reference to ‘.*’ is ambiguous",err)
            #print("error 5: ambiguous reference\n")
            label = "error: reference to "
            index = err.find(label)
            index_begin = index + len(label)
            index_end =err.find("’",index_begin + 1)
            to_define_char = err[:index_end][index_begin + 1:]
            #print(to_define_char)
            correct_ambiguous(source_file,to_define_char)

        #elif err != "":
        #    print(err)
            #error_file.append(source_file)
            #error_file.append(err)
        '''

write_file("error_file.txt", error_file)
write_file("error.txt", error_list)
