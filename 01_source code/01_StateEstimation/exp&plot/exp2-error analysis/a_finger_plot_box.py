"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4]
labels = ['TH', 'FF', 'MF']
th = np.array([1.9861, 1.6691, 2.2397, 1.9393, 2.1241, 2.2805, 2.0551, 1.8606, 2.1690])
ff = np.array([1.6393, 1.4455, 1.8336, 1.5126, 1.5507, 1.6798, 1.4914, 1.5135, 1.5695])
mf = np.array([1.4494, 1.2291, 1.6377, 1.2418, 1.3782, 1.4748, 1.3331, 1.2807, 1.4700])

all_data = [th, ff, mf]

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

colors_pale = ['#7fc97f', '#beaed4', '#fdc086']

box = plt.boxplot(all_data, patch_artist=True, positions=[1, 2.5, 4],
                  showmeans=False, widths=0.8, sym="k+")
for patch, color in zip(box['boxes'], colors_pale):
    patch.set_facecolor(color)

plt.ylabel("Error (mm)", fontdict=axis_font)
plt.xlim(0, 5)
plt.ylim(0, 6)
plt.xticks(ticks=x, labels=labels)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.patch.set_facecolor("silver")  # 设置ax1区域背景颜色
ax.patch.set_alpha(0.3)  # 设置ax1区域背景颜色透明度
plt.savefig("07.png", format="png", dpi=600)
plt.savefig("07_long.svg", format="svg")
plt.show()
