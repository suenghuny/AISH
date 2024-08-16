import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 데이터 로드
with open("state feature0.json", encoding='utf-8') as f:
    data_memory = json.load(f)

# 데이터 준비 (100개 샘플만 사용)
X = np.array([item[4] for item in data_memory])[:5000]
y = np.array([item[-1] for item in data_memory])[:5000]  # NumPy 배열로 변환
print(X.shape)



tsne = tsne = TSNE(
    n_components=1,
    perplexity=70,
    learning_rate=200,
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
#scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.6)
scatter = plt.scatter(X_2d, y, alpha=0.6)

# 컬러바 추가
plt.colorbar(scatter)

plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('2D t-SNE visualization with Viridis color mapping')
plt.tight_layout()
plt.savefig('feature4.png')