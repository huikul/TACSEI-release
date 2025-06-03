"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4, 5.5, 7]
labels = ['MK1', 'MK2', 'MK3', 'MK4', 'MK5']
joint1 = np.array([0.7050, 0.6002, 0.6719, 0.5502, 0.8088, 0.4992])
joint2 = np.array([1.0097, 0.9288, 1.2690, 1.0007, 1.1868, 1.0490])
joint3 = np.array([1.6612, 1.3169, 1.9352, 1.6991, 1.8029, 1.6271])
joint4 = np.array([2.0799, 1.8640, 2.7580, 2.4009, 2.4469, 2.3899])
joint5 = np.array([2.9431, 2.5949, 3.6060, 3.2597, 3.1696, 3.3406])

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
plt.savefig("10.png", format="png", dpi=600)
plt.savefig("10_long.svg", format="svg")
plt.show()
