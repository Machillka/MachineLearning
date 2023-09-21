import numpy as np
import cv2
import CreationFunctional as cf
x = np.load('GeneticAlgorithm/EdgeLogoCreator/GeneSave/OptimalGene.npy', allow_pickle=True)
print(x)
cv2.imshow("test", x[-1].Decoder())
cv2.waitKey(0)
cv2.destroyAllWindows()