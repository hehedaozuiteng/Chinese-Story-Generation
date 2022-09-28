import json

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
        for i in data:
            str = "<extra_id_1>" + i['story'] + "\n"
            f.write(str)

def write_txt_file_target_text(filename,data):
    with open(filename, "w+",encoding='utf-8') as f:
        for i in data:
            str = i['story'] + "\n"
            f.write(str)

def trans_txt2json_file(filename_w,filename_r):
    with open(filename_w, "w+",encoding='utf-8') as wf,open(filename_r, "r",encoding='utf-8') as rf:
        for line in rf.readlines():
            dic = {}
            dic["story"] = line 
            str1 = str(dic)+"\n"
            str1 = str1.replace("\'","\"")
            wf.write(str1)


if __name__ == '__main__':
    data = load_file("./outgen/train.jsonl")
    # read the data from json file

    write_txt_file_target_text("./result_train.txt",data)
    # convert data to txt file 

    trans_txt2json_file("outgen/result.jsonl", "result.txt")
    # extract the story from jsonl file and save as txt file 

    write_txt_file_source("./train.source",data)
    # save the data to source file 
    write_txt_file_target("./train.target",data)
    # save the data to target file 


