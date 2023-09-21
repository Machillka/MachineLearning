import CreationFunctional as cf
import numpy as np
import cv2
import os

SIZE = 500
CELLCOUNT = 20                                           # 利用 5 个三角形 拟合 edge
GENELENGTH = 24
GENESIZE = 7
COORSIZE = (0, 800)
CANVASSIZE = (800, 800, 3)
BACKGROUNDCOLOR = 255

MAXSIM = 0.99                                         # 允许最小误差

VariationRate = 0.56
CrossRate = 0.83

GENEPATH = 'GeneSave/Population.npy'
SIMPATH = 'GeneSave/SIM.txt'
GENEMAXPATH = 'GeneSave/OptimalGene.npy'

edgeLogo = cv2.imread('../../Database/edgeLogo.png')

def SingleTrain():
    
    # --- 加载模型
    population = list(cf.PopulationLoader(GENEPATH))
    newPopulation = []

    for single in population:
        if np.random.rand() < CrossRate:
            parent = population[np.random.randint(SIZE)]
            single = cf.Cross(single, parent, GENESIZE, CELLCOUNT)
        if np.random.rand() < VariationRate:
            single = cf.Variation(single, GENESIZE, CELLCOUNT)
        newPopulation.append(single)
    
    # population = population + newPopulation
    # fitness = cf.GetFitnessAll(population, edgeLogo)
    # chosenIndex = cf.ChoosePopulation(int(SIZE * 2), fitness, SIZE)
    fitness = cf.GetFitnessAll(newPopulation, edgeLogo)
    chosenIndex = cf.ChoosePopulation(SIZE, np.array(fitness), SIZE)

    pop = [newPopulation[i] for i in chosenIndex]
    fit = [fitness[i] for i in chosenIndex]

    sim = max(fit)
    index = fit.index(sim)
    with open(SIMPATH, 'a') as file:
        file.write(str(sim) + '\n')
    
    lastGene = []
    if os.path.exists(GENEMAXPATH):
        lastGene = list(np.load(GENEMAXPATH, allow_pickle = True))
    np.save(GENEMAXPATH, lastGene + [pop[index]])
    
    cf.PopulationSaver(pop, GENEPATH)

if __name__=='__main__':
    SingleTrain()