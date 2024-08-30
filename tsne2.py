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

# 데이터 준비
for dataset in [2]:
    X = np.array([item[dataset - 1] for item in data_memory])
    y = np.array([item[-1] for item in data_memory])
    tsne = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate=200,
        n_iter=2000,
        metric='euclidean',
        early_exaggeration=15,
        init='pca',
        method='barnes_hut',
        random_state=42
    )
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.6)

    cbar = plt.colorbar(scatter)
    cbar.set_label('V(s)', fontsize=24, labelpad=15)  # 'V(s)' 라벨 크기 증가
    cbar.ax.tick_params(labelsize=20)  # 컬러바 눈금 폰트 크기

    # 레이블 및 제목 설정 (글자 크기 증가)
    plt.xlabel('t-SNE feature 1', fontsize=24)
    plt.ylabel('t-SNE feature 2', fontsize=24)

    # 축 눈금 레이블 크기 설정
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.savefig('graph_feature{}.png'.format(dataset), dpi=600, bbox_inches='tight')
    plt.close()  # 메모리 관리를 위해 그림 객체 닫기