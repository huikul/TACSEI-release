"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4, 5.5, 7, 8.5, 10, 11.5, 13]
labels = ['1', '2', '4', '8', '16', '32', '64', '128', '256']

model_4    = np.array([4.7598, 4.0766, 5.3414, 4.5341, 4.4166, 4.5934, 6.175, 3.9814, 4.38939, 4.2062])
model_8    = np.array([4.6302, 4.4294, 4.2542, 4.2806, 4.3454, 4.3230, 4.2942, 4.2542, 4.3038, 4.3974])
model_16   = np.array([4.5228, 4.7348, 4.8012, 4.6684, 4.7884, 4.2124, 4.2908, 4.2388, 4.2788, 4.3308])
model_32   = np.array([4.1876, 4.2196, 4.2788, 4.2580, 4.2252, 4.1756, 4.1988, 4.1588, 4.1668, 4.1988])
model_64   = np.array([4.3725, 4.2789, 4.3437, 4.3365, 4.3533, 4.2981, 4.4469, 4.4069, 4.2901, 5.0012])
model_128  = np.array([4.7665, 4.6601, 4.3225, 4.4641, 4.5449, 4.8649, 4.5105, 4.3793, 4.3833, 5.4825])
model_256  = np.array([4.5387, 4.2955, 4.7267, 4.7819, 4.6563, 4.8907, 4.5419, 5.1155, 4.1563, 6.1363])
model_512  = np.array([6.8228, 6.6836, 4.6332, 5.7612, 5.7668, 4.8260, 4.2612, 4.9580, 4.8868, 5.7708])
model_1024 = np.array([5.7821, 6.1669, 5.1421, 5.1021, 5.5517, 7.0645, 6.3485, 4.4269, 4.8501, 5.4869])


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

plt.ylabel("Time (ms)", fontdict=axis_font)
plt.xlim(0, 14)
plt.ylim(1, 8.0)
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
