import numpy as np
import matplotlib.pyplot as plt
import random
#------ 数据集创建 ------#   PS: 西瓜数据集4.0
Dataset = [
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.637, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
    [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
    [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459],
]

# bufX, bufY = make_classification(n_samples = 1000, n_features = 2, n_informative = 2, n_redundant = 0, n_clusters_per_class = 1, random_state = 4)
# bufX = bufX.tolist()
# Dataset = [bufX[i] for i in range(len(bufX))]

#------ 超参数设定 ------#
K = 3

def distance(x1, x2, p = 2.0):
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    return np.power(np.sum((x1 - x2) ** p), float(1) / float(p))

def Calculate(cluster):
    layerSize = len(cluster[0])
    cluSize = len(cluster)

    sumLayer = [0.0] * layerSize
    for i in cluster:
        for j in range(layerSize):
            sumLayer[j] += i[j]    

    return list(np.array(sumLayer) / float(cluSize))

def JudgeSame(l1, l2):
    print(l1, l2)
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return False

    return True

def Train(data, k):
    # gather = [data[i] for i in range(k)]                               # 得到第一类
    gather = random.sample(data, k)
    cluster = {}

    for i in range(k):
        cluster[i] = list()

    running = True
    while running == True:
        plt.clf()
        cluster = {}
        for i in range(k):
            cluster[i] = list()

        for x in data:
            dist = [distance(gather[i], x) for i in range(k)]          # 得到距离
            idx = dist.index(min(dist))                                # 最小距离
            cluster[idx].append(x)

        #--- 计算新的均值向量
        gatherBuf = [Calculate(cluster[i]) for i in cluster]

        # print([data.index(x) for x in cluster[1]])

        if JudgeSame(gatherBuf, gather):
            running = False

        gather.clear()
        for i in gatherBuf:
            gather.append(i)

        Usemodle(cluster, gather)

    return cluster, gatherBuf

def Usemodle(cluster, mid):
    print('C:\n', cluster)
    print('Mid:\n', mid)

    color = ['r', 'g', 'b', 'y', 'm']

    for key in cluster: 
        plt.scatter([i[0] for i in cluster[key]], [i[1] for i in cluster[key]], c = color[key], marker = 'o')
        plt.scatter(mid[key][0], mid[key][1], c = color[key], marker = '+')
    
    plt.pause(0.5)  # 暂停0.5秒


cluster, mid = Train(Dataset, K)

plt.ioff()
plt.show()

