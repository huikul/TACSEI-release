"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4, 5.5, 7, 8.5, 10]
labels = ['TD', 'CP', 'AP', 'TD+\nCP', 'TD+\nAP', 'CP+\nAP', 'TD+\nCP+AP']
td = np.array([4.6044, 4.1944, 5.2334, 5.0595, 5.3739, 5.1732])
cp = np.array([3.8552, 3.7301, 3.4945, 3.5742, 3.2527, 3.4347])
ap = np.array([4.8251, 4.9900, 5.0780, 4.7864, 5.6986, 4.7975])
td_cp = np.array([2.9777, 3.2016, 3.0844, 3.6603, 3.4860, 3.2644, 3.1400, 3.8978, 3.2934])
td_ap = np.array([5.5544, 4.7937, 4.4549, 4.4667, 5.4186, 4.9559])
cp_ap = np.array([3.4004, 3.4822, 3.3647, 3.2800, 3.4586, 3.8120])
td_cp_ap = np.array([3.9783, 3.1432, 3.1508, 3.3728, 3.1291, 3.2789, 3.1913, 3.2572, 3.1393])

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
plt.savefig("06.png", format="png", dpi=600)
plt.savefig("06_long.svg", format="svg")
plt.show()
