import pickle
from keras.preprocessing import image
from tqdm import tqdm
from VGGFace.get_VGG_vector import get_VGG_vector



id_list = None
image_list = None
vectors_array = None



def get_image_names():
    global id_list
    id_list = []
    for i in range(1, person_num+1):
        person_id_str = str(i).zfill(2)
        # person_id_list = []
        for j in range(1,11):
            image_id_str = str(j).zfill(2)
            image_name = path + person_id_str + image_id_str + ".jpg"
            id_list.append(image_name)

def get_image():
    print("载入图片中...")
    global image_list
    image_list = []
    for i in tqdm(range(len(id_list))):
        im = image.load_img(
            id_list[i], 
            target_size=(224, 224))
        im = image.img_to_array(im)
        image_list.append(im)

def process_vectors():
    print("VGGFace网络处理中...")
    global vectors_array
    vectors_array =[]
    for i in tqdm(range(len(image_list))):
        vectors_array.append(
            get_VGG_vector(
                image_list[i]))

def save_vectors():
    with open(save_file_name, "wb") as file:
        pickle.dump(vectors_array, file)
    print("Done!")



def main():
    get_image_names()
    get_image()
    process_vectors()
    save_vectors()

if __name__ == "__main__":
    print("训练集")
    person_num = 60
    path = "./dataset/train/"
    save_file_name = "./dataset/VGGvectors/PubTrain.bin"
    main()

    print("测试集")
    person_num = 140
    path = "./dataset/eval/"
    save_file_name = "./dataset/VGGvectors/PubEval.bin"
    main()

    