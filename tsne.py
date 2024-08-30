import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

# Times New Roman 폰트 설정
plt.rcParams['font.family'] = 'Times New Roman'

# 데이터 로드 및 준비
with open("state feature final rl.json", encoding='utf-8') as f:
    data_memory = json.load(f)

dataset = 1
X = np.array([item[dataset-1] for item in data_memory])
y = np.array([item[-1] for item in data_memory])
print(X.shape)

# t-SNE 적용
tsne = TSNE(n_components=1, perplexity=20, learning_rate=200, n_iter=2000,
            metric='euclidean', early_exaggeration=15, init='pca',
            method='barnes_hut', random_state=42)
X_1d = tsne.fit_transform(X)

# 데이터프레임 생성
df = pd.DataFrame({'tsne_feature': X_1d.flatten(), 'V(s)': y})

# 각 고유한 tsne_feature 값에 대해 평균 V(s) 계산
avg_df = df.groupby('tsne_feature')['V(s)'].mean().reset_index()
avg_df = avg_df.sort_values('tsne_feature')

# 플롯 생성
plt.figure(figsize=(12, 8), facecolor='white')

# 산점도 그리기
scatter = plt.scatter(df['tsne_feature'], df['V(s)'], alpha=0.5, s=20,
                      color='#4C72B0', edgecolors='none')

# 평균선 그리기
plt.plot(avg_df['tsne_feature'], avg_df['V(s)'], color='#C44E52', linewidth=2)

# 레이블 및 제목 설정
plt.xlabel('t-SNE feature 1', fontsize=20)
plt.ylabel('V(s)', fontsize=20)
# plt.title('t-SNE visualization with average V(s) for feature {}'.format(dataset),
#           fontsize=16, fontweight='bold')

# 범례 추가
plt.legend(['Average V(s)', 'Data points'], loc='upper left', fontsize=12)

# 그리드 추가
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('state_feature{}_with_correct_average.png'.format(dataset),
            dpi=500, bbox_inches='tight')
plt.show()