import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


training = np.load("./KDD99/training_normal.npy")

scale = StandardScaler()
scaled_data = scale.fit_transform(training)


pca = PCA(n_components=2)
pca.fit(scaled_data)
print(pca.get_feature_names_out())


#plt.xlabel("Test x")
#plt.ylabel("Test y")

#plt.show()
