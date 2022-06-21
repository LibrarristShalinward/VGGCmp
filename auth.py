import argparse
from keras.preprocessing import image
from VGGFace.get_VGG_vector import get_VGG_vector
import numpy as np
from keras.models import load_model

parser = argparse.ArgumentParser(description='图片对比')
 
parser.add_argument('--im1', "-i2", default = "im1.jpg")
parser.add_argument('--im2', "-i1", default = "im2.jpg")
args = parser.parse_args()

im1 = image.load_img(args.im1, target_size=(224, 224))
im2 = image.load_img(args.im2, target_size=(224, 224))
im1 = image.img_to_array(im1)
im2 = image.img_to_array(im2)
x = np.array(get_VGG_vector(im1) + get_VGG_vector(im2))
x = np.expand_dims(x, axis = 0)

model = load_model("./authen/G3.h5")
y = model.predict(x)[0]

print("匹配度为%.4f%%" %(y[0] * 100.))
print("认证成功" if y[0] > y[1] else "认证失败")
