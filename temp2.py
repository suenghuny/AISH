import matplotlib.pyplot as plt
import numpy as np

# 데이터
data = {
    52: [0.0, 0.0, 0.0, 0.0, 0.13636363636363635, 0.3636363636363637, 0.18181818181818182, 0.0, 0.0, 0.0, 0.0, 0.0],
    110: [0.0, 0.0, 0.0, 0.0, 0.13636363636363635, 0.45454545454545464, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0],
    126: [0.0, 0.045454545454545456, 0.09090909090909091, 0.27272727272727276, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    254: [0.0, 0.0, 0.09090909090909091, 0.045454545454545456, 0.2272727272727273, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    118: [0.0, 0.0, 0.09090909090909091, 0.18181818181818182, 0.3636363636363637, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    247: [0.0, 0.0, 0.09090909090909091, 0.2272727272727273, 0.09090909090909091, 0.2272727272727273, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0],
    82: [0.0, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    280: [0.045454545454545456, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    88: [0.0, 0.0, 0.0, 0.09090909090909091, 0.18181818181818182, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    143: [0.0, 0.0, 0.0, 0.09090909090909091, 0.18181818181818182, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    66: [0.0, 0.0, 0.09090909090909091, 0.18181818181818182, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    253: [0.0, 0.0, 0.09090909090909091, 0.18181818181818182, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

colors = {
    52: '#FFA500', 110: '#FFA500',  # 주황색
    126: '#FFD700', 254: '#FFD700',  # 금색
    118: '#32CD32', 247: '#32CD32',  # 라임그린
    82: '#DC143C', 280: '#DC143C',  # 크림슨
    88: '#4169E1', 143: '#4169E1',  # 로얄블루
    66: '#8A2BE2', 253: '#8A2BE2'  # 블루바이올렛
}

# 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


def create_histogram(key, values, color, ax):
    bars = ax.bar(range(len(values)), values, color=color, edgecolor='black', linewidth=1.5, alpha=0.8)

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, 0.5)
    ax.set_xlim(-0.5, len(values) - 0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.set_title(f'Key: {key}', fontsize=14, fontweight='bold', pad=15)


# 데이터를 2개씩 그룹화
keys = list(data.keys())
grouped_data = [(keys[i], keys[i + 1]) for i in range(0, len(keys), 2)]

for i, (key1, key2) in enumerate(grouped_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    create_histogram(key1, data[key1], colors[key1], ax1)
    create_histogram(key2, data[key2], colors[key2], ax2)

    fig.suptitle('Histogram Comparison', fontsize=20, fontweight='bold', y=1.05)
    plt.tight_layout()

    plt.savefig(f'histograms_pair_{i + 1}.png', dpi=600, bbox_inches='tight')
    plt.close()

print("모든 히스토그램 쌍이 저장되었습니다.")