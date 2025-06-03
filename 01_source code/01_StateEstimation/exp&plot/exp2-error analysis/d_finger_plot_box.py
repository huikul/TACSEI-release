"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4]
labels = ['TH', 'FF', 'MF']
th = np.array([4.0071, 4.6423, 4.0226, 3.9004, 4.2630, 4.7765])
ff = np.array([3.4617, 3.1986, 2.8011, 2.8231, 3.2162, 3.5531])
mf = np.array([2.8340, 3.0124, 2.5109, 2.5139, 3.2514, 2.9185])

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
plt.savefig("13.png", format="png", dpi=600)
plt.savefig("13_long.svg", format="svg")
plt.show()
