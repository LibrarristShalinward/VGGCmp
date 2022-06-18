from keras.layers import Dense, Input, Dropout
from keras.models import Model
# from keras.optimizers import Adam
import numpy as np
from data_get import get_data
from time import time
import datetime as d

file_name = "./authen/" + d.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_mlp.h5"
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
        epochs = 10, 
        batch_size = 128)

    p = lambda x: model.predict(x)

    acc1 = evaluate(xt, yt, p)
    acc2 = evaluate(xe, ye, p)
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
    begin = time()
    plc = train()
    print("\nRunning Time: %.2fs\n" %(time() - begin))

    plc.save(file_name)