import numpy as np
import cv2
from skimage.metrics import structural_similarity

# 构建 chromosome 结构
class Chromosome:
    '''geneLength: 染色体长度'''
    def __init__(self, geneLength, geneSize):
        self.geneLength = geneLength
        self.geneSize = geneSize
        self.gene = np.random.randint(2, size = (self.geneSize, self.geneLength))

# 创建 个体 结构
class Creation:
    def __init__(self, cellCount, geneLength, geneSize, coorRange, canvasSize, backgroundColor):
        self.creationSet = []
        self.translated = []
        self.cellCount = cellCount
        self.geneLength = geneLength
        self.geneSize = geneSize
        self.coorRange = coorRange
        self.canvasSize = canvasSize
        self.backgroundColor = backgroundColor

        for i in range(cellCount):
            self.creationSet.append(Chromosome(geneLength, geneSize))
    
        self.canvas = np.ones((800, 800, 3), np.uint8) * backgroundColor


    '''把染色体解码出来 ( 翻译成性状 )'''
    def Decoder(self):
        # canvasR, canvasG, canvasB = [np.ones(self.canvasSize) * self.backgroundColor] * 3
        # canvas = np.ones(self.canvasSize) * self.backgroundColor
        canvas = np.ones((800, 800, 3), np.uint8) * self.backgroundColor
        for gene in self.creationSet:
            intCoor = gene.gene[:-1].dot(2 ** np.arange(self.geneLength)) / (1 << self.geneLength) * (self.coorRange[1] - self.coorRange[0])
            
            # 切割成 8bit
            color = []
            for i in range(0, self.geneLength, 8):
                color.append(gene.gene[-1, i : i + 8].dot(2 ** np.arange(8)))
            color = tuple(int(c) for c in color)
            # 转化坐标
            coor = []
            for i in range(0, self.geneSize - 1, 2):
                coor.append((int(intCoor[i]), int(intCoor[i + 1])))

            canvas = cv2.fillConvexPoly(canvas, points=np.array([coor]), color=color)
            
        self.canvas = canvas

        return canvas

    def GetFitness(self, realImg):
        return structural_similarity(self.Decoder(), realImg, channel_axis = 2)
    
'''返回索引'''
def ChoosePopulation(populationSize, fitnessAll, chooseCount):
    idx = np.random.choice(np.arange(populationSize), p = fitnessAll / sum(fitnessAll), size = chooseCount, replace = True)
    return idx

'''传入个体, 进行变异'''
def Variation(creation, geneSize, cellCount):
    left = np.random.randint(geneSize / 2)
    right = np.random.randint(geneSize / 2, geneSize)
    line = np.random.randint(0, geneSize)
    for switchGene in range(cellCount):
        for i in range(left, right):
            creation.creationSet[switchGene].gene[line][i] = creation.creationSet[switchGene].gene[line][i] ^ 1
    return creation

'''传入两个个体, 用 b 替换 a'''
def Cross(a, b, geneSize, cellCount):
    line = np.random.randint(0, geneSize)
    left = np.random.randint(geneSize / 2)
    right = np.random.randint(geneSize / 2, geneSize)
    for switchGene in range(cellCount):
        a.creationSet[switchGene].gene[line][left : right] = b.creationSet[switchGene].gene[line][left : right]
    return a 

def GetFitnessAll(population, realImg):
    fitness = []
    for single in population:
        # fitness.append(np.exp(single.GetFitness(realImg)))
        fitness.append(single.GetFitness(realImg))
    return fitness

def PopulationSaver(population, path):
    s = []
    for single in population:
        s.append(single)
    s = np.array(s)
    np.save(path, s)

def PopulationLoader(path):
    s = np.load(path, allow_pickle = True)
    return s   