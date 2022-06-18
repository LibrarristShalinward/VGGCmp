from keras_vggface.vggface import VGGFace
# 获取VGGFace网络
# Based on VGG16 architecture -> old paper(2015)
vggface = VGGFace(model='vgg16', include_top = False) # or VGGFace() as default

vggface.summary()

vggface.save("origin_VGG_Face.hd5")