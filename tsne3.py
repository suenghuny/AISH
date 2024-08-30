import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from matplotlib import font_manager

plt.rcParams['font.family'] = 'Times New Roman'

# 데이터 로드
with open("graph feature final rl.json", encoding='utf-8') as f:
    data_memory = json.load(f)
with open("state feature final rl.json", encoding='utf-8') as f:
    state_memory = json.load(f)

# 데이터 준비
dataset = 3
X = np.array([item[dataset - 1] for item in data_memory])
y = np.array([item[-1] for item in data_memory])

tsne = TSNE(
    n_components=2,
    perplexity=20,
    learning_rate=200,
    n_iter=2000,
    metric='cosine',
    early_exaggeration=15,
    init='pca',
    method='barnes_hut',
    random_state=42
)

X_2d = tsne.fit_transform(X)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.6)
cbar = plt.colorbar(scatter)
cbar.set_label('V(s)', fontsize=24, labelpad=15)
cbar.ax.tick_params(labelsize=20)

# 특정 점들을 색상으로 표시하고 라벨 추가
highlight_indices = [52, 110, 126, 254, 118, 247, 82, 280, 88, 143, 66, 253]
color_map = {
    52: '#FFA500', 110: '#FFA500',  # 주황색
    126: '#FFD700', 254: '#FFD700',  # 금색
    118: '#32CD32', 247: '#32CD32',  # 라임그린
    82: '#DC143C', 280: '#DC143C',  # 크림슨
    88: '#4169E1', 143: '#4169E1',  # 로얄블루, 블루바이올렛
    66: '#8A2BE2', 253: '#8A2BE2'   # 도저블루, 미디엄퍼플
}

for idx in highlight_indices:
    color = color_map[idx]
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=color, s=100, zorder=5)
    plt.annotate(str(idx), (X_2d[idx, 0], X_2d[idx, 1]), xytext=(5, 5),
                 textcoords='offset points', fontsize=12, color=color, weight='bold')

plt.xlabel('t-SNE feature 1', fontsize=24)
plt.ylabel('t-SNE feature 2', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.savefig('graph_feature{}_highlighted_color3_annotes.png'.format(dataset), dpi=600, bbox_inches='tight')
plt.close()