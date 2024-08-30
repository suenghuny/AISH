import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯을 위해 필요합니다

# Times New Roman 폰트 설정
plt.rcParams['font.family'] = 'Times New Roman'

# 데이터 로드
with open("state feature2.json", encoding='utf-8') as f:
    data_memory = json.load(f)

# 데이터 준비 (5000개 샘플 사용)
dataset = 6
X = np.array([item[dataset-1] for item in data_memory])[:5000]
y = np.array([item[-1] for item in data_memory])[:5000]
print(X.shape)

# t-SNE 적용
tsne = TSNE(
    n_components=3,
    perplexity=70,
    learning_rate=200,
    n_iter=2000,
    metric='cosine',
    early_exaggeration=4,
    init='pca',
    method='barnes_hut',
    random_state=42
)
X_3d = tsne.fit_transform(X)

# 3D 플롯 생성
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 3D 산점도 그리기
scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap='viridis', alpha=0.6)

# 컬러바 추가
cbar = plt.colorbar(scatter)

# 레이블 및 제목 설정
ax.set_xlabel('t-SNE feature 1', fontsize=14)
ax.set_ylabel('t-SNE feature 2', fontsize=14)
ax.set_zlabel('t-SNE feature 3', fontsize=14)
ax.set_title('3D t-SNE visualization for feature {}'.format(dataset), fontsize=16)

plt.tight_layout()
plt.savefig('feature{}_3d.png'.format(dataset), dpi=300, bbox_inches='tight')
plt.show()