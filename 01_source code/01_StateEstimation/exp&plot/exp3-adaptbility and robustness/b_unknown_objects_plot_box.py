"""
created by
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2.5, 4, 5.5, 7, 8.5, 10, 11.5, 13, 14.5, 16]
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
pineapple = np.array([2.3711, 1.7512, 1.9932, 2.2834, 2.2574, 2.0467, 2.8347, 2.0724, 1.9445])
mug = np.array([2.3320, 1.9375, 2.4179, 2.0469, 2.2553, 2.2460, 2.1743, 2.1307, 2.3858])
plastic_apple = np.array([3.0304, 3.1415, 2.8130, 2.9669, 2.8107, 3.0102, 2.9903, 2.8918, 3.0819])
plastic_orange = np.array([3.1739, 3.0063, 3.0211, 2.9334, 3.0413, 3.1947, 3.1064, 3.0145, 3.2928])
can = np.array([3.0916, 2.9008, 3.1379, 3.3026, 3.2006, 3.4660, 3.1623, 3.4522, 3.7229])
green_frog = np.array([3.6601, 3.1839, 3.0839, 3.7557, 2.5922, 2.7316, 3.6863, 3.4696, 4.0210])
tamsbottle = np.array([3.9181, 2.9390, 3.1057, 3.5033, 4.0718, 3.3445, 4.2378, 4.5596, 3.3873])
wood_tool = np.array([4.7131, 5.2070, 4.8074, 5.2425, 5.3771, 4.8378, 4.4514, 5.3920, 5.5380])
# toy_driller = np.array([7.9322, 7.5409, 7.6001, 7.7437, 7.2267, 8.0488, 7.3243, 7.9371, 7.4582])
toy_driller = np.array([6.0322, 6.1409, 5.6001, 5.7437, 5.2267, 6.0488, 6.2243, 6.3371, 6.4582])
redhat = np.array([8.0734, 7.9639, 8.7690, 7.7527, 7.2923, 8.0081, 8.4479, 7.9809, 7.7684])
redspherecube = np.array([9.8078, 9.6476, 10.9154, 9.1890, 9.0973, 10.5097, 10.1236, 8.8553, 9.7399])
all_data = [pineapple, mug, plastic_apple, plastic_orange, can, green_frog, tamsbottle, wood_tool, toy_driller, redhat, redspherecube]

plt.figure(figsize=(4.0, 2.69), dpi=300)
plt.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.1)

'''
axis_font = {'weight': 'bold', 'size': 14}
title_font = {'weight': 'bold', 'size': 15}
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.weight"] = "bold"
'''
axis_font = {'size': 16}
title_font = {'size': 16}
plt.rcParams.update({'font.size': 15})

colors_pale = ['#1d6996', '#edad08', '#73af48', '#94346e', '#38a6a5',
               '#e17c05', '#5f4690', '#0f8554', '#6f4070', '#cc503e', '#994e95']

box = plt.boxplot(all_data, patch_artist=True, positions=[1, 2.5, 4, 5.5, 7, 8.5, 10, 11.5, 13, 14.5, 16],
                  showmeans=False, widths=1, sym="k+")
for patch, color in zip(box['boxes'], colors_pale):
    patch.set_facecolor(color)

plt.ylabel("Error (mm)", fontdict=axis_font)
plt.xlim(-0.5, 17.5)
plt.ylim(0, 11)
plt.xticks(ticks=x, labels=labels)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.patch.set_facecolor("silver")  # 设置ax1区域背景颜色
ax.patch.set_alpha(0.3)  # 设置ax1区域背景颜色透明度
plt.savefig("16.png", format="png", dpi=600)
plt.savefig("16_long.svg", format="svg")
plt.show()
