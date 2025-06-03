"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4]
labels = ['TH', 'FF', 'MF']
th = np.array([2.1767, 1.9991, 1.9092, 2.0038, 2.1383, 1.9277, 1.6668, 1.8194])
ff = np.array([2.2578, 1.7360, 2.1080, 1.9048, 1.8350, 1.7682, 2.1037, 1.7633])
mf = np.array([2.0265, 1.6987, 1.7462, 2.0167, 1.9341, 1.8260, 1.5767, 1.6886])

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
plt.savefig("11.png", format="png", dpi=600)
plt.savefig("11_long.svg", format="svg")
plt.show()
