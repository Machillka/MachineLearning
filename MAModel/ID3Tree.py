from math import log2
import numpy as np

#------ 创建数据集
# Dataset = [
#         ['长', '粗', '男'],
#         ['短', '粗', '男'],
#         ['短', '粗', '男'],
#         ['长', '细', '女'],
#         ['短', '细', '女'],
#         ['短', '粗', '女'],
#         ['长', '粗', '女'],
#         ['长', '粗', '女']
# ]
# Attribute = ['头发','声音']
# UseAttribute =  ['头发','声音']                                                                               
# classification = ['男', '女']

Dataset = [
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],  
    ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
    ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
    ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
    ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
    ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
    ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
]
Attribute = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
UseAttribute =  ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']                                                                               
classification = ['好瓜', '坏瓜']                                # 分类结果

# /*===== 函数设定 ======*/
''' Well done '''
def CutData(data, key):                                         # 按照分类依据分出子集  key 传入的是属性对应值 
    cutdata = []
    
    for idx in data:
        if key in idx:
            cutdata.append(idx)

    return cutdata

''' Well done '''
def GetValue(data, idx):                                        # 根据属性获得取值     
    return set([x[idx] for x in data])

''' Well done '''
def CutDown(data, idx):                                         # 根据 idx 删除值
    cut = []
    for i in data:
        buf = i[: idx] + i[idx + 1:] 
        cut.append(buf)
    return cut



''' Well done '''
def Countlist(data):
    countLables = {}                                            # 计算分类后正反类对应的数量

    for key in classification:
        countLables[key] = 0

    for idx in data:            
        countLables[idx[-1]] += 1
    
    return countLables

#------ 计算信息熵
''' Well done '''
def Ent(data):
    datasize = len(data)                                        # 集合里的数据条数
    countLables = Countlist(data)                               # 计算数据里正反类对应的数量

    sum = 0
    for key in countLables:
        buf = float(countLables[key]) / float((datasize)) 
        if buf != 0:
            sum -= buf * log2(buf)

    return sum

''' Well done '''
def Gain(data, lable):                                          # 计算信息增益
    baseEnt = Ent(data)
    dataSize = len(data)
    axis = Attribute.index(lable)                               # 得到标签对应索引
    
    countKeys = list(GetValue(data, axis))                      # 得到标签对应属性取值      

    for i in data:
        if i[axis] not in countKeys:
            countKeys.append(i[axis])
    
    for i in countKeys:
        child = CutData(data, i)                                # 得到子集   
        baseEnt -= float(len(child)) / float(dataSize) * Ent(child)

    return baseEnt

def Choose(data, lables):                                       # 选择分类属性
    compare = [Gain(data, i) for i in lables]
    idx = compare.index(max(compare))
    return lables[idx], idx

def CreateTree(data, lables):
    countLables = Countlist(data)                               # 得到此数据正反例数量
    countData = [i[-1] for i in data]
    
    #--- /* ====== case 1 ====== */ ---#
    if countData.count(countData[0]) == len(countData):
        return countData[0]

    #--- /* ====== case 2 ====== */ ---#
    if len(data[0]) == 1:                                       # 数据集里只有一个元素 (分类结果)
        return max(countLables, key = lambda x: countLables[x])

    #--- /* ====== case 3 ====== */ ---#
    chooseFeature_name, chooseFeature_idx = Choose(data, lables)# 得到划分属性 
    decisionTree = {chooseFeature_name:{}}                      # 根节点  

    del(lables[chooseFeature_idx])                              # 删除父节点分类标签
    values = GetValue(data, chooseFeature_idx)                  # 得到子节点取值
    for v in values:
        cutLables = lables[:]
        decisionTree[chooseFeature_name][v] = CreateTree(CutDown(CutData(data, v), chooseFeature_idx), cutLables)
    return decisionTree
 
def UseModle(input, decisionTree):                              # 使用
    for i in decisionTree:
        idx = UseAttribute.index(i)
        values = GetValue(Dataset, idx)                         # 得到所有属性        
        for j in values:
            if input[idx] == j:
                if isinstance(decisionTree[i][j], dict):
                    UseModle(input, decisionTree[i][j])
                else:
                    print(decisionTree[i][j])

UseModle(['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑'], CreateTree(Dataset, Attribute))