import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
#print(vs.shape)
from cfg import get_cfg
cfg = get_cfg()


with open('visual feature final rl.json', 'r') as file:
    data_memory = json.load(file)

print(np.array(data_memory).shape)
discr_n = np.array(data_memory).shape[-1]
data_memory = np.array(data_memory).reshape(-1, discr_n)

vs = data_memory.mean(axis=0)
x = np.arange(discr_n)
plt.figure(figsize=(10, 6))
plt.plot(x, vs, marker='o', color='blue')


with open('visual feature final random.json', 'r') as file:
    data_memory = json.load(file)

print(np.array(data_memory).shape)
discr_n = np.array(data_memory).shape[-1]
data_memory = np.array(data_memory).reshape(-1, discr_n)

vs = data_memory.mean(axis=0)
plt.plot(x, vs, marker='o', color='green')


with open('visual feature final random2.json', 'r') as file:
    data_memory = json.load(file)

print(np.array(data_memory).shape)
discr_n = np.array(data_memory).shape[-1]
data_memory = np.array(data_memory).reshape(-1, discr_n)

vs = data_memory.mean(axis=0)
plt.plot(x, vs, marker='o', color='red')

# 그래프 꾸미기
plt.title('(12,) 모양 어레이의 꺾은선 그래프')
plt.xlabel('인덱스')
plt.ylabel('값')
plt.grid(True)

# 그래프 표시
plt.show()