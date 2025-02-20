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


#plt.show()
plt.xlabel("Test x")
plt.ylabel("Test y")
plt.grid(True)
plt.scatter(testingNormalPCA[:, 0], testingNormalPCA[:, 1])

plt.show()

class KMeans:
    '''
    The value k represents the number of clusters that will be made, the default given is 3
    The centroids haven't been randomly initialized yet and as such the variable has been left empty for now
    '''
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))
    
    def fit(self, X, max_iter=100):
        #Initializes the centroids in random places within the min/max of the data points
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for _ in range(max_iter):
            #Y represents the cluster labels for the dataset
            Y = []

            #Adds data points to clusters
            for data_point in X:
                distance = KMeans.euclidean_distance(data_point, self.centroids)
                cluster = np.argmin(distance)
                Y.append(cluster)
            
            Y = np.array(Y)

            cluster_idxs = []

            for i in range(self.k):
                cluster_idxs.append(np.argwhere(Y==i))

            cluster_centers = []

            #Reorganize cluster centroids
            for i, idx in enumerate(cluster_idxs):
                if len(idx) == 0:
                    #No change to centroid
                    cluster_centers.append(self.centroids[i])
                else:
                    #Centroid has been shifted using the mean position of all data points in the cluster
                    cluster_centers.append(np.mean(X[idx], axis=0)[0])

            #Breaks the loop early in there is no noticeable change in the centroid locations, otherwise save the new locations to the centroid attribute
            if np.max(self.centroids - np.array(cluster_centers)) < 0.001:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return Y
    
def main():
    random_test = np.random.randint(0, 100, (100, 2))
    kmeans = KMeans(k=5)
    labels = kmeans.fit(random_test)

    plt.scatter(random_test[:, 0], random_test[:, 1], c=labels)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)), marker="*", s=200)

    plt.show()

main()

