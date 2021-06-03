# import pandas as pd
import numpy as np
import geohash as gh
import os
import requests

#data = geohash.encode(116.255421,40.201209)
#print(data)

# 读取转换后的文件
#data = pd.read_csv(r'Data1101/no_stay_point.csv',low_memory=False) # header = None 不要列名
# 打印文件 前五行
#print(data)
# print(data.head(5))
#print(data[['LONGITUDE','LATITUDE']]) # 读取多列要两个中括号

# 数据筛选 只要纬度为104.0726的数据
#data = data[data['LONGITUDE'] == 104.0726]
# data = data[['LONGITUDE','LATITUDE']]
# 只把经纬度写入新文件
# data.to_csv(r'newData.csv', index = None, encoding = 'utf-8')
#print(data)
# data = pd.read_csv(r'newData.csv', low_memory=False)
# data_list=[]
# latitude=data["LATITUDE"]
# longitude=data["LONGITUDE"]
# for i in range(1,54158):
#     geohash = gh.encode(float(latitude[i - 1]), float(longitude[i - 1]))  # geohash将经纬度编码
#     data_list.append([latitude[i - 1], longitude[i - 1], geohash])  # 将经纬度和geohash码放到数组中
# df = pd.DataFrame(data_list, columns=['latitude', 'longitude', 'geohash'])  # 给数据对应的列名
# df.to_csv(r'geohashR11.csv', index=None)  # 保存生成的csv文件

# data = pd.read_csv(r'newData.csv', low_memory=False)
# a = data.values.reshape(-1,2)
# #csv文件读取进来的数据并非矩阵格式。需要进行转换
# print(a)

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from geographiclib.geodesic import Geodesic

# -*- coding: UTF-8 -*-
# 计算数据点两两之间的距离
def getDistanceMatrix(datas):
    N, D = np.shape(datas)
    dists = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            vi = datas[i, :]
            vj = datas[j, :]
            dists[i, j] = np.sqrt(np.dot((vi - vj), (vi - vj)))
    return dists


# 找到密度计算的阈值dc
# 要求平均每个点周围距离小于dc的点的数目占总点数的1%-2%
def select_dc(dists):
    '''算法1'''
    N = np.shape(dists)[0]
    tt = np.reshape(dists, N * N)
    percent = 2.0
    position = int(N * (N - 1) * percent / 100)
    dc = np.sort(tt)[position + N]

    ''' 算法 2 '''
    # N = np.shape(dists)[0]
    # max_dis = np.max(dists)
    # min_dis = np.min(dists)
    # dc = (max_dis + min_dis) / 2

    # while True:
    # n_neighs = np.where(dists<dc)[0].shape[0]-N
    # rate = n_neighs/(N*(N-1))

    # if rate>=0.01 and rate<=0.02:
    # break
    # if rate<0.01:
    # min_dis = dc
    # else:
    # max_dis = dc

    # dc = (max_dis + min_dis) / 2
    # if max_dis - min_dis < 0.0001:
    # break
    return dc


# 计算每个点的局部密度
def get_density(dists, dc, method=None):
    N = np.shape(dists)[0]
    rho = np.zeros(N)

    for i in range(N):
        if method == None:
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho


# 计算每个数据点的密度距离
# 即对每个点，找到密度比它大的所有点
# 再在这些点中找到距离其最近的点的距离
def get_deltas(dists, rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)
    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            continue

        # 对于其他的点
        # 找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离,并找最小值
        deltas[index] = np.min(dists[index, index_higher_rho])

        # 保存最近邻点的编号
        index_nn = np.argmin(dists[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber


# 通过阈值选取 rho与delta都大的点
# 作为聚类中心
# def find_centers_auto(rho, deltas):
#     rho_threshold = (np.min(rho) + np.max(rho)) / 2
#     delta_threshold = (np.min(deltas) + np.max(deltas)) / 2
#     N = np.shape(rho)[0]
#
#     centers = []
#     for i in range(N):
#         if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
#             centers.append(i)
#     return np.array(centers)


# 选取 rho与delta乘积较大的点作为
# 聚类中心
def find_centers_K(rho, deltas, K):
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return centers[:K]


def cluster_PD(rho, centers, nearest_neiber):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return

    N = np.shape(rho)[0]
    labs = -1 * np.ones(N).astype(int)

    # 首先对几个聚类中进行标号
    for i, center in enumerate(centers):
        labs[center] = i

    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 从密度大的点进行标号
        if labs[index] == -1:
            # 如果没有被标记过
            # 那么聚类标号与距离其最近且密度比其大
            # 的点的标号相同
            labs[index] = labs[int(nearest_neiber[index])]
    return labs


def draw_decision(rho, deltas, name="0_decision.jpg"):
    plt.cla()
    for i in range(np.shape(datas)[0]):
        plt.scatter(rho[i], deltas[i], s=16., color=(0, 0, 0))
        plt.annotate(str(i), xy=(rho[i], deltas[i]), xytext=(rho[i], deltas[i]))
        plt.xlabel("rho")
        plt.ylabel("deltas")
    plt.savefig(name)


def draw_cluster(datas, labs, centers, dic_colors, name="0_cluster.jpg"):
    plt.cla()
    K = np.shape(centers)[0]

    for k in range(K):
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        # 画数据点
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color=dic_colors[k])
        # 画聚类中心
        plt.scatter(datas[centers[k], 0], datas[centers[k], 1], color="k", marker="+", s=200.)
    plt.savefig(name)


if __name__ == "__main__":
    dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0),
                  2: (0, 0, .8), 3: (.8, .8, 0),
                  4: (.8, 0, .8), 5: (0, .8, .8),
                  6: (0, 0, 0)}
    data = pd.read_csv(r'newData01.csv', low_memory=False)
    datas = data.values.reshape(-1, 2)
    # csv文件读取进来的数据并非矩阵格式。需要进行转换
    # print(datas)

    # 计算距离矩阵
    dists = getDistanceMatrix(datas)
    # 计算dc
    dc = select_dc(dists)
    print("dc", dc)
    # 计算局部密度
    rho = get_density(dists, dc, method="Gaussion")
    # 计算密度距离
    deltas, nearest_neiber = get_deltas(dists, rho)

    # 绘制密度/距离分布图
    draw_decision(rho, deltas, name= "_decision.jpg")

    # 获取聚类中心点
    centers = find_centers_K(rho, deltas, 3)
    # centers = find_centers_auto(rho,deltas)
    print("centers", centers)

    labs = cluster_PD(rho, centers, nearest_neiber)
    draw_cluster(datas, labs, centers, dic_colors, name= "_cluster.jpg")

