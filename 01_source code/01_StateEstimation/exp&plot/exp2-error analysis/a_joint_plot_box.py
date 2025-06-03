"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4, 5.5, 7]
labels = ['MK1', 'MK2', 'MK3', 'MK4', 'MK5']
joint1 = np.array([2.5247/3, 2.2255/3, 0.7147, 0.6433, 0.6005, 0.6945, 0.6296, 0.9786])
joint2 = np.array([3.3834/3, 3.0197/3, 0.9886, 1.0870, 1.0527, 1.1074, 1.1089, 1.2098])
joint3 = np.array([4.5920/3, 3.6788/3, 1.3913, 1.5343, 1.6010, 1.4779, 1.3583, 1.5108])
joint4 = np.array([6.1970/3, 5.1559/3, 1.9598, 2.1750, 2.2993, 2.0024, 1.8095, 2.0943])
joint5 = np.array([8.6766/3, 7.6390/3, 2.7683, 2.9820, 3.5050, 2.8505, 2.8518, 2.8873])

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
plt.savefig("08.png", format="png", dpi=600)
plt.savefig("08_long.svg", format="svg")
plt.show()
