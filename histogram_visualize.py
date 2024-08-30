import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import json

def plot_histogram(al):
    # JSON 파일에서 데이터 읽어오기
    with open(f"history_{al}.json", "r", encoding='utf-8') as json_file:
        history = json.load(json_file)

    # Times New Roman 폰트 설정
    font_times = FontProperties(family='Times New Roman', size=12)

    # 색상 선택
    color_choice = 'green' if al == 'random' else 'blue'

    plt.figure(figsize=(10, 6))
    plt.hist(history, bins=100, edgecolor='black', density=True, color=color_choice, alpha=0.5)

    plt.xlabel('Distance', fontproperties=font_times, fontsize=14)
    plt.ylabel('Proportion', fontproperties=font_times, fontsize=14)

    # 그리드 추가
    plt.grid(True, linestyle='--', alpha=0.7)

    # y축을 비율로 표시 (0에서 1 사이)
    plt.ylim(0, plt.ylim()[1])
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

    # x축과 y축의 눈금 레이블에 폰트 적용
    plt.xticks(fontproperties=font_times)
    plt.yticks(fontproperties=font_times)

    # 그래프 저장
    plt.savefig(f"{al}_histogram_proportion.png", dpi=600, bbox_inches='tight')
    plt.close()

# 각각의 히스토그램 그리기
plot_histogram('random')
plot_histogram('rl')

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import json

# Times New Roman 폰트 설정
font_times = FontProperties(family='Times New Roman', size=12)

# JSON 파일에서 데이터 읽어오기
with open("history_random.json", "r", encoding='utf-8') as json_file:
    history_random = json.load(json_file)

with open("history_rl.json", "r", encoding='utf-8') as json_file:
    history_rl = json.load(json_file)

plt.figure(figsize=(10, 6))

# 두 데이터셋을 같은 그래프에 그리기
plt.hist(history_rl, bins=100, edgecolor='black', alpha=0.5, color='blue', label='RL')
plt.hist(history_random, bins=100, edgecolor='black', alpha=0.5, color='red', label='Random')


plt.xlabel('Distance', fontproperties=font_times, fontsize=14)
plt.ylabel('Count', fontproperties=font_times, fontsize=14)

# 그리드 추가
plt.grid(True, linestyle='--', alpha=0.7)

# x축과 y축의 눈금 레이블에 폰트 적용
plt.xticks(fontproperties=font_times)
plt.yticks(fontproperties=font_times)

# 범례 추가
plt.legend(prop=font_times)

# 그래프 저장
plt.savefig("combined_histogram_count.png", dpi=600, bbox_inches='tight')
plt.close()