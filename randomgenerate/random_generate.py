std = 1.56e+005
heavy = 1000
light = 7.8e+004

total = std + heavy+light
print(total)

import random
total_list = [i for i in range(100,int(total),100)]
random.seed(114514)
import numpy as np
rs_list = []
for i in range(100):
    a = random.randint(1000,int(total))
    b = random.randint(1000,int(total))
    if a > b:
        temp = a
        a = b
        b = temp
    temp_list = []
    temp_list.append(a)
    temp_list.append(b-a)
    temp_list.append(total-b)
    rs_list.append(temp_list)
np_rs = np.array(rs_list)
print(np_rs)
