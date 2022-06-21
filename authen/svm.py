from data_get import get_data
import numpy as np
from sklearn.svm import LinearSVC
from time import time

xt, yt, xe, ye = get_data()
n = 2622
policy = LinearSVC(C = 1, loss = "hinge")

xt_ = np.log(np.abs(xt[:, :n] - xt[:, n:]) + 1e-10)
yt_ = (yt[:, 0] == 1).astype(np.float64)

policy.fit(xt_, yt_)

begin = time()
xe_ = np.log(np.abs(xe[:, :n] - xe[:, n:]) + 1e-10)
yep = policy.predict(xe_)
acc = np.sum(np.abs(yep - ye[:, 0])) / len(ye)
print("\n运行时间: %.2fs\n" %(time() - begin))
print(acc)