import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
#print(vs.shape)
from cfg import get_cfg
cfg = get_cfg()

with open('visual feature final random2.json', 'r') as file:
    data_memory = json.load(file)

print(np.array(data_memory).shape)
data_memory = np.array(data_memory)
data_memory = data_memory[:, :60, :]

vs = data_memory.mean(axis=0)

# vs 데이터를 DataFrame으로 변환
df = pd.DataFrame(vs, columns=[i for i in range(cfg.discr_n)])

# 플롯 크기 설정
plt.figure(figsize=(12, 8))

# 데이터 준비
x = []
y = []
colors = []
for i in range(len(df)):
    for j in range(len(df.columns)):
        x.append(j)
        y.append(i)
        colors.append(df.iloc[i, j])

# 산점도 생성
scatter = plt.scatter(x, y, c=colors, cmap="turbo", s=100, vmin=0, vmax=0.008)

# 컬러바 추가
cbar = plt.colorbar(scatter)
cbar.set_label('미사일 분포', rotation=270, labelpad=15)

# 차트 꾸미기
plt.title('Random', fontsize=16)
plt.xlabel('구간', fontsize=12)
plt.ylabel('시간', fontsize=12)

# x축, y축 눈금 설정
plt.xticks(range(len(df.columns)), df.columns)
plt.yticks(range(len(df)), [f'{i + 1}' for i in range(len(df))])

# 격자 추가
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('random2.jpg')