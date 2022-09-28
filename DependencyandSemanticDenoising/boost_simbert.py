import json
import copy
import re
from roformer_sim.test.generate import gen_synonyms
from tqdm import trange


def load_file(filename, pred=False):
    data = []
    with open(filename, "r",encoding='utf-8') as f:
        for line in f.readlines():
            if pred:
                data.append({"label": line.strip()})
            else:
                data.append(json.loads(line))
        f.close()
    return data

def write_file(filename,datas):
    with open(filename, "w+", encoding='utf-8') as f:
        for data in datas:
            for i in data:
                string = str(i) + "\n"
                string = string.replace("\'", "\"")
                f.write(string)

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")

def maker_fn(text,n):
    return ["make"+text+"ddd"]*n

def boost_data(data,maker,n):
    datas = [data]
    for i in range(n):
        new_data = copy.deepcopy(data)
        datas.append(new_data)
    for i in trange(len(data)):
        for data_index in range(n):
            datas[data_index+1][i]['story'] = ""
        strs = cut_sent(data[i]['story'])
        for s in strs:
            sentances = maker(s,n=20,k=n)
            for data_index in range(n):
                datas[data_index + 1][i]['story'] = datas[data_index + 1][i]['story'] + sentances[data_index]
    return datas

def dumpling_data(data,n):
    datas = [data]
    for i in range(n):
        new_data = copy.deepcopy(data)
        datas.append(new_data)
    return datas

def write_txt_file_source(filename,datas):
    with open(filename, "w+",encoding='utf-8') as f:
        for data in datas:
            for i in data:
                str = i['title']
                for s in i['outline']:
                    str = str + "#" + s
                str = str + "<extra_id_1>\n"
                f.write(str)

def write_txt_file_target(filename,datas):
    with open(filename, "w+",encoding='utf-8') as f:
        for data in datas:
            for i in data:
                str = "<extra_id_1>" + i['story'] + "\n"
                f.write(str)

def write_jsonl_file_source(filename,data):
    datas = dumpling_data(data,len(data))
    with open(filename, "w+",encoding='utf-8') as f:
        for d in datas:
            str1 = str(d) + "\n"
            str1 = str1.replace("\'", "\"")
            f.write(str1)

if __name__ == '__main__':

    data = load_file("./boosts_bert/train.jsonl") # read the training data from json file
    data = boost_data(data, gen_synonyms, 5) # Expanded data to 6 times the original size
    write_txt_file_source("./boosts_bert/train.source", data) # save the outline from data to the file 
    write_txt_file_target("./boosts_bert/train.target", data) # save the story from data to the file 

    #write_jsonl_file_source("./boosts_bert/train.jsonl", data) #saving the data as jsonal file
