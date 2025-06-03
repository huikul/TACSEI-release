"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4, 5.5, 7, 8.5, 10]
labels = ['TD', 'CP', 'AP', 'TD+\nCP', 'TD+\nAP', 'CP+\nAP', 'TD+\nCP+AP']
td = np.array([3.8603, 4.1151, 3.6171, 3.5562, 4.0517, 3.6354])
cp = np.array([2.2956, 2.1037, 2.1403, 2.4286, 2.1853, 2.4149])
ap = np.array([4.2659, 4.1027, 4.0216, 4.0188, 4.2164, 4.2174])
td_cp = np.array([2.0640, 2.3962, 1.8604, 1.8479, 2.0335, 1.8179])
td_ap = np.array([2.9123, 2.8543, 2.8765, 4.4565, 2.8888, 2.4067])
cp_ap = np.array([1.7051, 1.8666, 1.9949, 1.7477, 1.8979, 1.9212])
td_cp_ap = np.array([1.7424, 1.7060, 1.6774, 1.9419, 1.7268, 1.7404])

all_data = [td, cp, ap, td_cp, td_ap, cp_ap, td_cp_ap]

plt.figure(figsize=(6.5, 4.0), dpi=300)
plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.13)

'''
axis_font = {'weight': 'bold', 'size': 14}
title_font = {'weight': 'bold', 'size': 15}
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.weight"] = "bold"
'''
axis_font = {'size': 16}
title_font = {'size': 16}
plt.rcParams.update({'font.size': 15})

colors_pale = ['#66c5cc', '#f6cf71', '#f89c74', '#dcb0f2', '#87c55f', '#9eb9f3', '#fe88b1']

box = plt.boxplot(all_data, patch_artist=True, positions=[1, 2.5, 4, 5.5, 7, 8.5, 10],
                  showmeans=False, widths=0.8, sym="k+")
for patch, color in zip(box['boxes'], colors_pale):
    patch.set_facecolor(color)

plt.ylabel("Error (mm)", fontdict=axis_font)
plt.xlim(0, 11)
plt.ylim(1.2, 6)
plt.xticks(ticks=x, labels=labels)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.patch.set_facecolor("silver")  # 设置ax1区域背景颜色
ax.patch.set_alpha(0.3)  # 设置ax1区域背景颜色透明度
plt.savefig("04.png", format="png")
plt.savefig("04_long.svg", format="svg")
plt.show()
