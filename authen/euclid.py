from data_get import get_data
import numpy as np

xt, yt, xe, ye = get_data()
n = 2622

d_j = []
for i, j in zip(xt, yt): 
    d = np.linalg.norm(i[:n] - i[n:])
    d_j.append([d, j[0]])

d_j = np.array(sorted(d_j, key = lambda dj: dj[0]))

template = d_j[:, 1] * 0
mn = len(d_j)
th = -1
for i in range(len(d_j)): 
    template[i] = 1
    loss = np.sum(np.abs(template - d_j[:, 1]))
    if loss < mn: 
        th = i
        mn = loss

threshold = (d_j[th, 0] + d_j[th + 1, 0]) / 2
print("欧氏距离方法阈值为%.4f" %(threshold))

error_c = 0
for i, j in zip(xe, ye): 
    d = np.linalg.norm(i[:n] - i[n:])
    judge = 1 if d < threshold else 0
    if j[0] != judge: 
        error_c += 1

print("欧氏距离方法准确率为%.2f%%" %((1 - error_c / len(xe)) * 100))
