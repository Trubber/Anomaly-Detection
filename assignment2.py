import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


training = np.load(r"D:\School\Machine Learning\KDD99\training_normal.npy")
testingNormal = np.load(r"D:\School\Machine Learning\KDD99\testing_normal.npy")
testingAttack = np.load(r"D:\School\Machine Learning\KDD99\testing_attack.npy")

scale = StandardScaler()
scaled_data = scale.fit_transform(training)


pca = PCA(n_components=2)
pca.fit(scaled_data)
print(pca.get_feature_names_out())

testingNormalScaled = scale.transform(testingNormal)
testingNormalPCA = pca.transform(testingNormalScaled)


plt.xlabel("Test x")
plt.ylabel("Test y")
plt.grid(True)
plt.scatter(testingNormalPCA[:, 0], testingNormalPCA[:, 1])

plt.show()
