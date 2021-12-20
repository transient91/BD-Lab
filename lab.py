#!/usr/bin/env python
# coding: utf-8

import random
from operator import add

import numpy as np
from pyspark import SparkContext


def is_ancestor(ancestor, descedant):
    if ancestor == descedant:
        return False
    for ch in descedant:
        if ch not in ancestor:
            return False
    return True


sc = SparkContext('local', 'lab')

data_filename = './data.txt'
data = sc.textFile(data_filename).map(lambda x: x.split(',')).map(
    lambda x: ((int(x[0]), x[1].replace('NONE', '')), int(x[-1])))
N = data.count()  # 多维视图格数

# sigma = .5 * (N - 1)
sigma = .5  # niche size

pc = 0.8  # 交叉概率
pm = 0.1  # 变异概率

k = 8  # 选择Top-K视图
vn = 10  #视图个数

iteration = 50

# TKV = [(3, 5, 6, 4), (4, 6, 7, 5), (4, 2, 5, 6), (7, 3, 6, 2), (3, 4, 6, 2),
#    (5, 7, 2, 3), (3, 5, 4, 2), (4, 2, 3, 7)]

# 初始随机Top-k-view
TKV = [
    tuple(np.random.choice(range(2, N), k, replace=False)) for _ in range(vn)
]

for it in range(iteration):
    # 计算目标函数值C_MV and C_NMV
    lst = []
    for i, tkv in enumerate(TKV):
        mv = data.filter(lambda x: x[0][0] in (*tkv, N))  # 根一定要物化
        c_mv = mv.values().sum()
        nmv = data.filter(lambda x: x[0][0] not in (*tkv, N)).map(
            lambda x: x[0][-1]).collect()
        c_nmv = 0
        for v in nmv:
            c_nmv += mv.filter(
                lambda x: is_ancestor(x[0][-1], v)).values().min()
        lst.append((i, tkv, (c_mv, c_nmv)))
    rdd_obj = sc.parallelize(lst)

    # 打印当前种群
    # print(it)
    # rdd_obj.foreach(print)

    with open('./output.txt', 'a+', encoding='UTF-8') as f:
        f.write('{}\n{}\n'.format(it,
                                  '\n'.join(str(i)
                                            for i in rdd_obj.collect())))

    # 求各个view的rank
    rdd_rank = sc.parallelize([])
    for item in rdd_obj.collect():
        obj = item[-1]
        rdd_tmp = rdd_obj.filter(
            lambda x: (x[-1][0] >= obj[0] and x[-1][1] > obj[1]) or
            (x[-1][0] > obj[0] and x[-1][1] >= obj[1])).map(lambda x: (x, 1))
        rdd_rank = sc.union([rdd_rank, rdd_tmp])
    rdd_rank = sc.union([rdd_rank, rdd_obj.map(lambda x: (x, 1))])
    rdd_rank = rdd_rank.reduceByKey(lambda a, b: a + b).sortByKey().cache()

    # rank为k的个数num
    rdd_num = rdd_rank.map(lambda x: (x[-1], 1)).reduceByKey(
        lambda a, b: a + b).cache()
    # rank, num

    fit_list = []
    for i, kv in enumerate(rdd_num.collect()):
        tmp = np.sum(rdd_num.map(lambda x: x[-1]).take(i))
        fit_list.append((kv[0], N - tmp - .5 * (kv[-1] - 1)))

    rdd_fit = sc.parallelize((fit_list)).join(rdd_num)
    # rank, fitness, num

    cmv_max = rdd_obj.map(lambda x: x[-1][0]).max()
    cnmv_max = rdd_obj.map(lambda x: x[-1][-1]).max()
    cmv_min = rdd_obj.map(lambda x: x[-1][0]).min()
    cnmv_min = rdd_obj.map(lambda x: x[-1][-1]).min()

    cmv_dist = cmv_max - cmv_min
    cnmv_dist = cnmv_max - cnmv_min

    nc_list = []
    for rank_n in rdd_num.collect():
        rank, n = rank_n

        share_array = np.eye(n)

        rdd_dist = rdd_rank.filter(lambda x: x[-1] == rank).map(
            lambda x: x[0][-1])

        obj_list = rdd_dist.collect()
        for i in range(n):
            for j in range(n):
                if j > i or i == j: break
                tmp = ((obj_list[i][0] - obj_list[j][0]) / cmv_dist)**2 + (
                    (obj_list[i][-1] - obj_list[j][-1]) / cnmv_dist)**2
                if tmp**.5 < .5:
                    share_array[i, j] = share_array[j, i] = 1 - tmp**.5 / .5

        nc_list += (list(np.sum(share_array, axis=1)))

    NC_list = []
    rdd_tmp = rdd_rank.sortBy(lambda x: x[-1]).map(lambda x: (x[-1], x[0][0]))

    rdd_tmp = rdd_tmp.join(rdd_fit).map(lambda x: (x[1][0], x[0], *x[1][1]))
    # seq, rank, fitness, num

    for i in range(vn):
        NC_list.append((*rdd_tmp.collect()[i], nc_list[i]))

    rdd_nc = sc.parallelize(NC_list).sortByKey().cache()
    # seq, rank, fitness, num, niche count

    rdd_fitt = rdd_nc.map(lambda x: (*x, x[2] / x[-1]))
    # seq, rank, fitness, num, niche count, fitness'

    rdd_tmp = rdd_fitt.map(lambda x: (x[1], x[-1])).reduceByKey(add)
    # 分母

    rdd_temp = rdd_tmp.join(rdd_fitt.map(lambda x: (x[1], x))).map(
        lambda x: tuple(reversed(x[-1]))).sortByKey()
    # seq, rank, fitness, num, niche count, fitness', 分母

    rdd_sfit = rdd_temp.map(
        lambda x: (x[0][0], (x[0][2] * x[0][3] * x[0][5]) / x[-1])).sortBy(
            keyfunc=(lambda x: x[-1]), ascending=False)

    sum = rdd_sfit.map(lambda x: x[-1]).sum()
    p_list = rdd_sfit.sortByKey().map(lambda x: x[1] / sum).collect()

    select_list = []
    for j in range(vn):
        ps = random.random()
        pt = p_list[0]
        for i in range(1, vn):
            if ps > pt:
                pt += p_list[i]
            else:
                break
        select_list.append(i - 1)
    mating_pool = []
    for i in select_list:
        mating_pool.append(list(TKV[i]))

    for i in range(0, vn, 2):
        if random.random() < pc:
            cp = random.randint(0, k - 1)
            # 保证Top-k视图互不相同
            if mating_pool[i][cp] not in mating_pool[i + 1] and mating_pool[
                    i + 1][cp] not in mating_pool[i]:
                tmp = mating_pool[i][cp]
                mating_pool[i][cp] = mating_pool[i + 1][cp]
                mating_pool[i + 1][cp] = tmp

    for i in range(vn):
        if random.random() < pm:
            mp = random.randint(0, k - 1)
            changeto = random.randint(2, N)
            # 保证Top-k视图互不相同
            if changeto not in mating_pool[i]:
                mating_pool[i][mp] = changeto

    TKV = [tuple(i) for i in mating_pool]
