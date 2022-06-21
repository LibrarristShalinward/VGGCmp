from PIL import Image
import csv
import requests
from os import remove

ration = 1.5

def download2folder(file_name: str = "./csvs/dev_urls.csv", target_folder: str = "./train/"): 
    name_list={} # 照片下载计数

    with open(file_name, "r") as f:
        r = csv.reader(f)
        for row in r:
            name = row[0]
            if name == "person": continue
            names = list(name_list.keys())
            if name in names:
                if name_list[name] > 10:
                    continue # 每人仅下载10张照片
                else:
                    name_index = names.index(name) + 1
                    image_index = name_list[name]
            else:
                name_index = len(name_list) + 1
                image_index = 1
            jpg_name = target_folder + str(name_index).zfill(2) + str(image_index).zfill(2) + ".jpg" # 图片文件名
            url = row[2]
            # 尝试下载
            try:
                reply = requests.get(url)
                reply.keep_alive = False
                if reply.status_code == 200:
                    with open(jpg_name, "ab") as im:
                        im.write(reply.content)
                else: 
                    print("Download error!!!")
                    #print("URL: " + url)
                    continue
                try:
                    # 尝试根据数据集切割图片
                    box = eval(row[3])
                    center = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
                    radius = int((center[0] - box[0]) * ration)
                    box = (
                        center[0] - radius,
                        center[1] - radius, 
                        center[0] + radius, 
                        center[1] + radius)
                    origin = Image.open(jpg_name)
                    scaled = origin.crop(box).resize((128, 128))
                    scaled.save(jpg_name, "JPEG")
                    print("Download " + jpg_name + " succeed!")
                    if name in names:
                        name_list[name] += 1
                    else:
                        name_list.update({name:2})
                except:
                    print("Download blank!!!")
                    remove(jpg_name)
                    #print("URL: " + url)
                    pass
            except:
                print("Download failed!!!")
                #print("URL: " + url)
                pass

if __name__ == "__main__": 
    download2folder() # 下载训练集
    download2folder(file_name = "./csvs/eval_urls.csv", target_folder = "./eval/") # 下载测试集