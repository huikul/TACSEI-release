"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 4, 7, 10, 13]
labels = ['DT1', 'DT2', 'DT3', 'DT4', 'All']
normal_grasping = np.array([1.4465, 1.5763, 1.4200, 1.9256, 1.7053, 1.7106, 1.6114, 1.5636, 1.6059])
changing_pressure = np.array([2.0299, 1.6713, 1.7325, 1.8779, 1.7597, 1.7936, 1.7968, 1.5681, 1.6326])
random_pressure = np.array([1.9580, 2.0728, 1.7505, 1.9519, 1.9572, 1.7776, 1.7648, 1.9544, 1.8982])
extra_disturbance = np.array([3.4367, 3.3103, 3.3009, 3.6123, 2.8939, 3.4513, 3.2989, 3.4994, 3.1770])
all = np.array([1.7302, 1.7871, 2.1433, 1.8442, 1.7925, 1.9606, 1.8363, 1.7837, 1.6587])

all_data = [normal_grasping, changing_pressure, random_pressure, extra_disturbance, all]

plt.figure(figsize=(6.5, 3.75), dpi=300)
plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08)

'''
axis_font = {'weight': 'bold', 'size': 14}
title_font = {'weight': 'bold', 'size': 15}
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.weight"] = "bold"
'''

axis_font = {'size': 16}
title_font = {'size': 16}
plt.rcParams.update({'font.size': 15})
# plt.xlabel("Sub dataset", fontdict=axis_font)

colors_pale = ['#1d6996', '#edad08', '#73af48', '#94346e', '#38a6a5']

box = plt.boxplot(all_data, patch_artist=True, positions=[1, 4, 7, 10, 13],
                  showmeans=False, widths=1.2, sym="k+")
for patch, color in zip(box['boxes'], colors_pale):
    patch.set_facecolor(color)

plt.ylabel("Error (mm)", fontdict=axis_font)
plt.xlim(-0.5, 14.5)
plt.ylim(0, 4)
plt.xticks(ticks=x, labels=labels)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.patch.set_facecolor("silver")  # 设置ax1区域背景颜色
ax.patch.set_alpha(0.3)  # 设置ax1区域背景颜色透明度
plt.savefig("15.png", format="png", dpi=600)
plt.savefig("15_long.svg", format="svg")
plt.show()
