import pickle


def get_data(): 
    with open("./dataset/VGGvectors/train.bin", "rb") as f: 
        xt, yt = pickle.load(f)
    with open("./dataset/VGGvectors/eval.bin", "rb") as f: 
        xe, ye = pickle.load(f)
    return xt, yt, xe, ye