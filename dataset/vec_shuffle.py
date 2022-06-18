from random import randint, random
import numpy as np
import pickle

from tqdm import tqdm

def get_shuffle_idx(person_num = 60, set_num = 7500, same_ration = .4): 
    idx_stack = []

    rand_index = lambda: randint(0, person_num - 1) * 10 + randint(0, 9)
    rand_same_index = lambda index: index - index % 10 + randint(0, 9)

    def rand_diff_index(index):
        a = randint(0, person_num - 1)
        while a == int(index // 10): a = randint(1, person_num - 1)
        return a * 10 + randint(0, 9)

    def rand_same_indexes(): 
        index = rand_index() 
        return [index, rand_same_index(index)]

    def rand_diff_indexes(): 
        index = rand_index() 
        return [index, rand_diff_index(index)]

    print("生成随机索引...")
    for _ in tqdm(range(set_num)):
        if random() < same_ration:
            index = rand_same_indexes()
            line = index + [1]
        else:
            index = rand_diff_indexes()
            line = index + [0]
        idx_stack.append(line)
    return idx_stack



def stack2set(dataset, stack): 
    print("生成数据集...")
    xs = []
    ys = []
    for sf in tqdm(stack): 
        # print(sf)
        xs.append(dataset[sf[0]] + dataset[sf[1]])
        ys.append([sf[2], 1 - sf[2]])
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


if __name__ == "__main__": 
    idxes = get_shuffle_idx()
    with open("VGGvectors/PubTrain.bin", "rb") as f: 
        data = pickle.load(f)
    with open("VGGvectors/train.bin", "wb") as f: 
        pickle.dump(stack2set(data, idxes), f)
    
    idxes = get_shuffle_idx(140, 2500, .6)
    with open("VGGvectors/PubEval.bin", "rb") as f: 
        data = pickle.load(f)
    with open("VGGvectors/eval.bin", "wb") as f: 
        pickle.dump(stack2set(data, idxes), f)

        