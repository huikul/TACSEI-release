"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4, 5.5, 7, 8.5, 10]
labels = ['TD', 'CP', 'AP', 'TD+\nCP', 'TD+\nAP', 'CP+\nAP', 'TD+\nCP+AP']
td = np.array([4.1595, 4.0070, 4.8792, 4.2536, 4.4275, 3.7516])
cp = np.array([2.3782, 2.5687, 2.7118, 2.5965, 2.5807, 2.7016])
ap = np.array([4.5916, 4.8985, 4.9686, 4.6933, 4.7516, 4,3681])
td_cp = np.array([2.3473, 2.5497, 2.4823, 2.5113, 2.5869, 2.2912])
td_ap = np.array([4.1563, 3.0918, 4.7485, 3.8904, 4.8547, 3.9270])
cp_ap = np.array([2.0467, 2.0295, 1.9097, 1.8478, 2.0361, 2.0967])
td_cp_ap = np.array([1.8552, 1.8744, 1.8481, 1.7062, 2.0141, 1.7622])

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
plt.savefig("05.png", format="png")
plt.savefig("05_long.svg", format="svg")
plt.show()
