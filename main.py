import math
import os
import numpy as np
import pandas as pd

import config as conf
from ga import Ga
import matplotlib.pyplot as plt
import time


config = conf.get_config()

# ### 通过随机生成数据，然后保存到本地
# # 计算每两个城市直接距离
# def build_dist_mat(input_list):
#     n = config.city_num
#     dist_mat = np.zeros([n, n])
#     for i in range(n):
#         for j in range(i + 1, n):
#             d = input_list[i, :] - input_list[j, :]
#             # 计算点积
#             dist_mat[i, j] = np.dot(d, d)
#             dist_mat[j, i] = dist_mat[i, j] # 对称矩阵
#     return dist_mat
#
#
# # 加载数据
# if not os.path.exists('city_pos_list.npy'):
#     # 城市坐标
#     city_pos_list = np.random.rand(config.city_num, config.pos_dimension)
#     # 城市距离矩阵
#     city_dist_mat = build_dist_mat(city_pos_list)
#
#     np.save('city_pos_list',city_pos_list) #将数组以二进制格式保存到磁盘，默认加上.npy后缀
#     np.save('city_dist_mat',city_dist_mat) #将数组以二进制格式保存到磁盘，默认加上.npy后缀
# else:
#     city_pos_list = np.load('city_pos_list.npy') #读取磁盘上的数组
#     city_dist_mat = np.load('city_dist_mat.npy') #读取磁盘上的数组
#
#

# ###

## 通过读取数据文件数据
def load_data(file_path):
    """
    读取数据文件
    :param file_path: 数据文件路径
    :return: 坐标列表
    """
    dataframe = pd.read_csv(file_path, sep=" ", header=None)
    v = dataframe.iloc[:, 1:3]
    # v = v/10000
    return np.array(v)

def calculate_distance_matrix(train_v):
    """
    计算城市间距离矩阵
    :param train_v: 坐标列表
    :return: 城市间距离矩阵
    """
    train_d = train_v
    dist = np.zeros((train_v.shape[0], train_d.shape[0]))
    for i in range(train_v.shape[0]):
        for j in range(train_d.shape[0]):
            dist[i, j] = math.sqrt(np.sum((train_v[i, :] - train_d[j, :]) ** 2))
    return dist
points = load_data("D:\Projects\ga-tsp-main\data\TSP100cities.tsp")

city_dist_mat = calculate_distance_matrix(points)
print(city_dist_mat)



# 遗传算法运行
start = time.time() # 测试中间代码块运行时间

ga = Ga(city_dist_mat)
result_list, fitness_list = ga.train()
result = result_list[-1]
#result_pos_list = city_pos_list[result, :]
result_pos_list = city_dist_mat[result, :]

print(fitness_list)
print('最小距离：', min(fitness_list))

end = time.time()
print('总用时：', end-start,'秒')
# 绘图方案1
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

fig = plt.figure()
plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r')
plt.title(u"路线")
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(fitness_list)
plt.title(u"适应度曲线")
plt.legend()
fig.show()

# 绘图方案2
# 解决中文显示问题
# plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
#
# # 根据结果绘图
# fig = plt.figure()
# x = result_pos_list[:, 0].copy().tolist()
# y = result_pos_list[:, 1].copy().tolist()
# np.savetxt("data.txt", result_pos_list)
# print("x轴", x)
# print("y轴", y)
# [:, 0]表示将二维数组的第一个下标全部取出并保存为一维数组 这里对应每个初始X轴的坐标
# plt.plot(x, y, 'o-r', label="路线")
# for a, b in zip(x, y):  # 添加这个循环显示坐标
#     a = round(a, 3)
#     b = round(b, 3)
#     plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)
# plt.title(u"路线")
# plt.legend()
# fig.show()
# plt.savefig("./route.png")
# plt.clf()
# fig = plt.figure()
# plt.plot(fitness_list, label="适应度")
# plt.title(u"适应度曲线")
# plt.legend()
# plt.savefig("./fitness.png")