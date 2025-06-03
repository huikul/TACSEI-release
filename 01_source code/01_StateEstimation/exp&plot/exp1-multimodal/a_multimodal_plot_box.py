"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4, 5.5, 7, 8.5, 10]
labels = ['TD', 'CP', 'AP', 'TD+\nCP', 'TD+\nAP', 'CP+\nAP', 'TD+\nCP+AP']
td = np.array([3.1815, 3.8090, 3.1864, 2.6574, 3.4908, 6.2786])
cp = np.array([1.9299, 1.9914, 2.0838, 2.0409, 2.0450, 2.1135])
ap = np.array([4.0951, 4.1213, 4.2876, 3.9625, 4.1627, 4.0472])
td_cp = np.array([1.7648, 1.8718, 1.6884, 1.5738, 1.5914, 2.0349])
td_ap = np.array([2.4628, 2.8427, 2.4603, 2.7078, 3.8474, 2.8586])
cp_ap = np.array([2.0157, 1.8409, 1.7422, 1.8789, 1.6897, 1.7789])
td_cp_ap = np.array([1.4932, 1.4470, 1.4638, 1.6658, 1.8735, 1.8050])

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
plt.savefig("03.png", format="png")
plt.savefig("03_long.svg", format="svg")
plt.show()
