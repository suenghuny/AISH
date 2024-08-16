import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import font_manager

# Times New Roman 폰트 설정
# font_path = "path/to/times new roman.ttf"  # 실제 폰트 파일 경로로 변경하세요
# font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'

# 데이터 로드
with open("state feature2.json", encoding='utf-8') as f:
    data_memory = json.load(f)

# 데이터 준비 (100개 샘플만 사용)
dataset = 6

X = np.array([item[dataset-1] for item in data_memory])[:5000]
y = np.array([item[-1] for item in data_memory])[:5000]  # NumPy 배열로 변환
print(X.shape)



tsne = tsne = TSNE(
    n_components=2,
    perplexity=50,
    learning_rate=50,
    n_iter=2000,
    metric='cosine',
    early_exaggeration=4,
    init='pca',
    method='barnes_hut',
    random_state=42
)
X_2d = tsne.fit_transform(X)
plt.figure(figsize=(10, 8))

# 산점도 그리기 (viridis 컬러맵 사용)
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.6)
#scatter = plt.scatter(X_2d, y, color='#FF7F00', alpha=0.6, s=30)

# 컬러바 추가
cbar = plt.colorbar(scatter)
#cbar.ax.tick_params(labelsize=14)  # 컬러바 눈금 폰트 크기

# 레이블 및 제목 설정
plt.xlabel('t-SNE feature 1', fontsize=16)
plt.ylabel('t-SNE feature 2', fontsize=16)
plt.title('2D t-SNE visualization for feature {}'.format(dataset))
plt.tight_layout()
plt.savefig('feature{}.png'.format(dataset))