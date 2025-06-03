"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4, 5.5, 7, 8.5, 10, 11.5, 13]
labels = ['1', '2', '4', '8', '16', '32', '64', '128', '256']
model_4 = np.array([1.1759, 1.0905, 1.2486, 1.1477, 1.1330, 1.1551, 1.3528, 1.0786, 1.1296, 1.1067])
model_8 = np.array([1.1227, 1.0976, 1.0757, 1.0790, 1.0871, 1.0843, 1.0807, 1.0757, 1.0819, 1.0936])
model_16 = np.array([1.1262, 1.1527, 1.1610, 1.1444, 1.1594, 1.0874, 1.0972, 1.0907, 1.0957, 1.1022])
model_32 = np.array([1.0493, 1.0533, 1.0607, 1.0581, 1.0540, 1.0478, 1.0507, 1.0457, 1.0467, 1.0507])
model_64 = np.array([1.0900, 1.0783, 1.0864, 1.0855, 1.0876, 1.0807, 1.0993, 1.0943, 1.0797])
model_128 = np.array([1.0468, 1.0585, 1.0538, 1.0590, 1.0566, 1.0466, 1.0523, 1.0859, 1.0864, 1.2238])
model_256 = np.array([1.0991, 1.1062, 1.0976, 1.1045, 1.1138, 1.0931, 1.1370, 1.2087, 1.0888, 1.3363])
model_512 = np.array([1.4200, 1.4026, 1.1463, 1.0373, 1.0380, 1.0454, 1.0998, 1.0619, 1.0530, 1.0385])
model_1024 = np.array([1.0843, 1.1324, 1.2543, 1.1243, 1.3055, 1.4946, 1.1551, 1.1649, 1.2178, 1.2974])

all_data = [model_4, model_8, model_16, model_32, model_64, model_128, model_256, model_512, model_1024]

plt.figure(figsize=(6.5, 4.0), dpi=300)
plt.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.12)

axis_font = {'weight': 'bold', 'size': 14}
title_font = {'weight': 'bold', 'size': 15}
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.weight"] = "bold"

colors_pale = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec', '#f2f2f2']

box = plt.boxplot(all_data, patch_artist=True, positions=[1, 2.5, 4, 5.5, 7, 8.5, 10, 11.5, 13],
                  showmeans=False, widths=0.8, sym="k+")
for patch, color in zip(box['boxes'], colors_pale):
    patch.set_facecolor(color)

plt.ylabel("Time (ms)", fontdict=axis_font)
plt.xlim(0, 14)
plt.ylim(1, 1.5)
plt.xticks(ticks=x, labels=labels)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.patch.set_facecolor("silver")  # 设置ax1区域背景颜色
ax.patch.set_alpha(0.3)  # 设置ax1区域背景颜色透明度
plt.savefig("02.png", format="png")
plt.savefig("02_long.svg", format="svg")
plt.show()
