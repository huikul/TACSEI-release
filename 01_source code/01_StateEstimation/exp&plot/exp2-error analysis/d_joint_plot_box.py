"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4, 5.5, 7]
labels = ['MK1', 'MK2', 'MK3', 'MK4', 'MK5']
joint1 = np.array([1.2118, 1.0257, 1.5666, 1.4157, 1.7712, 1.0397])
joint2 = np.array([2.1076, 2.6975, 2.1784, 3.1811, 2.1961, 2.0783])
joint3 = np.array([3.2479, 3.5869, 3.2801, 3.9042, 3.9728, 3.0409])
joint4 = np.array([3.8298, 4.8805, 4.4697, 4.5164, 4.4240, 4.0361])
joint5 = np.array([5.1605, 5.8981, 5.6766, 5.7295, 5.5203, 5.2006])

all_data = [joint1, joint2, joint3, joint4, joint5]

plt.figure(figsize=(4.5, 3.0), dpi=300)
plt.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.10)
'''
axis_font = {'weight': 'bold', 'size': 14}
title_font = {'weight': 'bold', 'size': 15}
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.weight"] = "bold"
'''
axis_font = {'size': 16}
title_font = {'size': 16}
plt.rcParams.update({'font.size': 15})

colors_pale = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#7cadee']

box = plt.boxplot(all_data, patch_artist=True, positions=[1, 2.5, 4, 5.5, 7],
                  showmeans=False, widths=0.8, sym="k+")
for patch, color in zip(box['boxes'], colors_pale):
    patch.set_facecolor(color)

plt.ylabel("Error (mm)", fontdict=axis_font)
plt.xlim(0, 8)
plt.ylim(0, 6)
plt.xticks(ticks=x, labels=labels)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.patch.set_facecolor("silver")  # 设置ax1区域背景颜色
ax.patch.set_alpha(0.3)  # 设置ax1区域背景颜色透明度
plt.savefig("14.png", format="png", dpi=600)
plt.savefig("14_long.svg", format="svg")
plt.show()
