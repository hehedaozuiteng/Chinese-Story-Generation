import hanlp
import json

hanlp.pretrained.mtl.ALL

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)

import re

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")

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


token_list_1 = ["nsubj","root","dobj","pobj"]


token_list = ["nsubj","root","dobj"]

def get_token(text):
    text = cut_sent(text)
    doc = HanLP(text, tasks='dep')
    strs = ""
    for toks , deps in zip(doc["tok/fine"],doc["dep"]):
        for tok , dep in zip(toks,deps):
            if dep[1] in token_list:
                strs = strs + tok + "<"+dep[1]+">"
            else:
                strs = strs + tok
    return strs

def write_txt_file_source(filename,data):
    with open(filename, "w+",encoding='utf-8') as f:
        for i in data:
            str = i['title']
            for s in i['outline']:
                str = str + "#" + s
            str = str + "<extra_id_1>\n"
            f.write(str)

def write_txt_file_target(filename,data):
    with open(filename, "w+",encoding='utf-8') as f:
        count = 0
        for i in data:
            count = count +1
            if count %10 == 0:
                print(count , len(data))
            truth = get_token(i['story'])
            str = "<extra_id_1>" + truth + "\n"
            f.write(str)

def trans_txt2json_file(filename_w,filename_r):
    with open(filename_w, "w+",encoding='utf-8') as wf,open(filename_r, "r",encoding='utf-8') as rf:
        for line in rf.readlines():
            dic = {}
            dic["story"] = line.replace("<nsubj>","").replace("<root>","").replace("<dobj>","").replace("<pobj>","")
            str1 = str(dic)+"\n"
            str1 = str1.replace("\'","\"")
            wf.write(str1)


def trans_txt2txt_file(filename_w,filename_r):
    with open(filename_w, "w+",encoding='utf-8') as wf,open(filename_r, "r",encoding='utf-8') as rf:
        for line in rf.readlines():
            str1 = line.replace("<nsubj>","").replace("<root>","").replace("<dobj>","").replace("<pobj>","")
            str1 = str1
            wf.write(str1)


"""
text = cut_sent(data[4]['story'])
doc = HanLP(text, tasks='dep')
doc.pretty_print()
"""

if __name__ == '__main__':

    data = load_file("./outgen/valid.jsonl")
    # read the training data from json file

    write_txt_file_source("./tsk_hanlp_nsubj/val.source",data)

    # adding the Dependency token into the story text in the data
    # and save the story text to the file

    write_txt_file_target("./tsk_hanlp_nsubj/val.target",data)
    # save the training target from data to the file 

