"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4, 5.5, 7, 8.5, 10, 11.5, 13]
labels = ['1', '2', '4', '8', '16', '32', '64', '128', '256']
model_4 = np.array([3.6337, 2.8860, 3.3588, 3.2742, 3.2340, 2.4191, 3.3639])
model_8 = np.array([2.0636, 2.7481, 3.0750, 2.9673, 2.3762, 3.4618, 3.1973])
model_16 = np.array([1.6264, 1.8603, 1.6966, 1.7966, 1.7080, 1.7884, 1.7371])
model_32 = np.array([1.7859, 1.8808, 1.6088, 1.7446, 1.7643, 1.4106, 1.9096])
model_64 = np.array([1.5941, 2.0465, 1.9237, 1.9092, 1.9501, 1.7500, 1.9777])
model_128 = np.array([1.8581, 1.4348, 1.8437, 1.8971, 1.9709, 1.8275, 1.9437])
model_256 = np.array([1.8719, 1.9279, 1.3673, 2.0682, 2.0714, 1.7444, 1.9762])
model_512 = np.array([2.4776, 2.3728, 2.1676, 1.6681, 1.7455, 2.5375, 2.8074])
model_1024 = np.array([2.1308, 1.5600, 6.2541, 5.4392, 1.8652, 2.0496, 1.8056])

all_data = [model_4, model_8, model_16, model_32, model_64, model_128, model_256, model_512, model_1024]

plt.figure(figsize=(6.5, 4.0), dpi=300)
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

colors_pale = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec', '#f2f2f2']

box = plt.boxplot(all_data, patch_artist=True, positions=[1, 2.5, 4, 5.5, 7, 8.5, 10, 11.5, 13],
                  showmeans=False, widths=0.8, sym="k+")
for patch, color in zip(box['boxes'], colors_pale):
    patch.set_facecolor(color)

plt.ylabel("Error (mm)", fontdict=axis_font)
plt.xlim(0, 14)
plt.ylim(1, 6.5)
plt.xticks(ticks=x, labels=labels)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.patch.set_facecolor("silver")  # 设置ax1区域背景颜色
ax.patch.set_alpha(0.3)  # 设置ax1区域背景颜色透明度
plt.savefig("01.png", format="png")
plt.savefig("01_long.svg", format="svg")
plt.show()
