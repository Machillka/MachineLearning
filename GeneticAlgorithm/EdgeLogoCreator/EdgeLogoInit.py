import CreationFunctional as cf
import os

# 初始化种群
SIZE = 500
CELLCOUNT = 20                                           # 利用 5 个三角形 拟合 edge
GENELENGTH = 24
GENESIZE = 7
COORSIZE = (0, 800)
CANVASSIZE = (800, 800, 3)
BACKGROUNDCOLOR = 255

VariationRate = 0.03
CrossRate = 0.8

GENEPATH = 'GeneSave/Population.npy'
SIMPATH = 'GeneSave/SIM.txt'
GENEMAXPATH = 'GeneSave/OptimalGene.npy'
population = []

for i in range(SIZE):
    population.append(cf.Creation(CELLCOUNT, GENELENGTH, GENESIZE, COORSIZE, CANVASSIZE, BACKGROUNDCOLOR))

if not os.path.exists("GeneSave/"):
    os.makedirs("GeneSave/")

if os.path.exists(GENEMAXPATH):
    os.remove(GENEMAXPATH)

# --- 保存初始化种群
cf.PopulationSaver(population, GENEPATH)

# --- 初始化 SIM 记录文件
with open(SIMPATH, 'w', encoding = 'utf-8') as simFile:
    print("Successfully Created!")