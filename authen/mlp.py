from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
# from keras.optimizers import Adam
import numpy as np
from data_get import get_data
from time import time
import datetime as d

file_name = "./authen/" + d.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_mlp.h5"
# mlp_struct = [2] # 网络O
# mlp_struct = [8, 2] # 网络G1
# mlp_struct = [16, 2] # 网络G2
# mlp_struct = [64, 2] # 网络G3
# mlp_struct = [256, 2] # 网络G4
# mlp_struct = [1024, 2] # 网络G5
# mlp_struct = [64, 64, 2] # 网络A1
# mlp_struct = [64, 64, 64, 2] # 网络A2
# mlp_struct = [64, 64, 64, 64, 2] # 网络A3
# mlp_struct = [64, 16, 2] # 网络B1
# mlp_struct = [64, 16, 16, 2] # 网络B2
# mlp_struct = [64, 16, 16, 16, 2] # 网络B3
mlp_struct = [1024, 256, 64, 16, 2]

def model_construct(input_size):
    x = Input(shape = (input_size, ))
    inter = x
    for num_unit in mlp_struct[:-1]:
        inter = Dense(units = num_unit, activation = "leaky_relu")(inter)
        inter = Dropout(.25)(inter)
    inter = Dense(units = mlp_struct[-1], activation = "softmax")(inter)
    model = Model(inputs = x, outputs = inter)
    model.compile(loss = "categorical_crossentropy", 
        optimizer = "adam", 
        metrics = ["accuracy"])
    model.summary()
    return model

def train(): 
    xt, yt, xe, ye = get_data()
    n = xt.shape[1]

    model = model_construct(n)

    model.fit(xt, yt, 
        validation_data = (xe, ye), 
        epochs = 50, 
        batch_size = 128, 
        callbacks = [EarlyStopping(monitor = "val_accuracy", patience = 3)])

    p = lambda x: model.predict(x)

    acc1 = evaluate(xt, yt, p)
    begin = time()
    acc2 = evaluate(xe, ye, p)
    print("\n运行时间: %.2fs\n" %(time() - begin))
    print("\nTraining Acc.: %.2f" %(acc1 * 100, ) + '%' + '\n')
    print("\nValidating Acc.: %.2f" %(acc2 * 100, ) + '%' + '\n')
    
    return model

def evaluate(x, y, p):
    n = len(x)
    re = p(x)

    return np.sum(((re[:, 0] - re[:, 1]) * (y[:, 0] - y[:, 1]) > 0).astype(np.float64)) / n

def cat2vec(cat_list):
    n = max(cat_list).item() + 1
    arr = np.zeros((len(cat_list), n))
    for i in range(len(cat_list)):
        arr[i][cat_list[i]] = 1.
    return arr

def vec2cat(arr):
    cat_list = []
    for line in arr:
        cat_list.append(
            list(line).index(max(line)))
    return cat_list

if __name__ == "__main__": 
    plc = train()

    plc.save(file_name)