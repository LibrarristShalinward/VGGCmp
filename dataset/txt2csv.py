# 将PubFig数据集提供的txt转换为csv，方便程序读取
import csv
import re



name_list = [
    "dev_people", 
    "dev_urls", 
    "eval_people", 
    "eval_urls"
    ] # 待处理txt列表
txt_path = "./PubFig/"
csv_path = "./csvs/"

def split_by_tab(str):
    # txt分割规则
    return re.split("\t", str)



def trans(filename):
    # 将txt转换为csv，方便爬虫程序读写
    txt_filename = txt_path + filename + ".txt"
    csv_filename = csv_path + filename + ".csv"
    with open(txt_filename,"r") as txt_file:
        with open(csv_filename, "w", newline='') as csv_file:
            writer = csv.writer(csv_file)
            lines = txt_file.readlines()
            list = split_by_tab(lines[1].strip('\n')) # 列标题
            writer.writerow(list[1:]) # 忽略行首注释符
            for line in lines[2:]:
                # 列内容
                list = split_by_tab(line.strip('\n'))
                writer.writerow(list)
    print("File " + filename + " finished!!!")



def trans1(filename):
    txt_filename = txt_path + filename + ".txt"
    csv_filename = csv_path + filename + "1.csv"
    name_dict = {}
    with open(txt_filename,"r") as txt_file:
        with open(csv_filename, "w", newline='') as csv_file:
            writer = csv.writer(csv_file)
            lines = txt_file.readlines()
            list = split_by_tab(lines[1].strip('\n')) # 列标题
            writer.writerow(list[1:]) # 忽略行首注释符
            for line in lines[2:]:
                list = split_by_tab(line.strip('\n'))
                name = list[0]
                if name in name_dict.keys():
                    if name_dict[name] > 0:
                        name_dict[name] -= 1
                        writer.writerow(list)
                else:
                    name_dict[name] = 49 # 每人仅保留50张图片
                    writer.writerow(list)
    print("File " + filename + "1 finished!!!")



if __name__ == "__main__": 
    for name in name_list:
        trans(name)
    for name in ["dev_urls", "eval_urls"]:
        trans1(name)

