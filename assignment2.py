import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

#Function used to graph data using data
def dataGraph(dataToScale, dataToGraph):
    
    finalData = PCAcalcs(dataToScale, dataToGraph)

    plt.grid(True)
    plt.scatter(finalData[:, 0], finalData[:, 1])

    plt.show()

#Function used to scale the data to 2D
def PCAcalcs(dataToScale, dataToConvert):
    scale = StandardScaler()
    scaled_data = scale.fit_transform(dataToScale)

    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    
    return pca.transform(scale.transform(dataToConvert))

class KMeans:
    '''
    The value k represents the number of clusters that will be made, the default given is 3
    The value of t represents the distance threshold between a data point and it's assigned cluster's centroid, data points outside the threshold are considered anomalous
    The centroids haven't been randomly initialized yet and as such the variable has been left empty for now
    The metrics parameter just sets whether the metrics will be printed at the end of the fit method (switch was created for hyperparameter testing)
    '''
    def __init__(self, k=3, t=0.5, metrics=True):
        self.k = k
        self.t = t
        self.centroids = None
        self.metrics = metrics

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))
    
    def fit(self, X, max_iter=100):
        #Initializes the centroids in random places within the min/max of the data points
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        #True Positive, False Positive, True Negative, False Negative
        tp= 1
        fp = 1
        tn = 1
        fn = 1  

        for _ in range(max_iter):
            #Y represents the cluster labels for the dataset
            Y = []

            #Adds data points to clusters
            for data_point in X:
                distance = KMeans.euclidean_distance(data_point, self.centroids)
                cluster = np.argmin(distance)
                #print(np.argmin(distance))
                #Checks the distance between data_point and it's closest centroid, it's anomalous if the distance is greater than the threshold
                if np.argmin(distance) < self.t:
                    tn += 1
                else:
                    tp += 1

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

               


        #Performance Metrics
        
        tpr = tp / (fn + tp)
        fpr = fp / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = 2*tp / (2*tp + fp + fn)
        if self.metrics:
            print("KMeans Metrics")
            print("TPR = " + str(tpr))
            print("FPR = " + str(fpr))
            print("Accuracy = " + str(accuracy))
            print("F-1 Score = " + str(f1_score))
            print("")
        

        return Y
    

class DBSCAN:
    '''
    # A DBSCAN class that contains the functions to preform a DBSCAN given specific data\n
    Epsilon: Default value of .5, This is used to determine how far from a core point is part of a given cluster\n
    minPoints: Default value of 5, Used to determine the minimum amount of neighbors within range to determine if a given point is a core point\n
    labels: Given Labels to help keep track of specific
    '''
    def __init__(self, epsilon=0.5, minPoints=5):
        self.epsilon = epsilon 
        self.minPoints = minPoints
        self.labels = None  

    def getLabels(self):
        #Just a function to get labels
        return self.labels

    @staticmethod
    def getNeighbors(self, dataPoint, data):

        neighbors = []

        for i, d in enumerate(data):
            #Finds the distance from the given dataPoint
            distance = np.linalg.norm(dataPoint - d)

            #Checks the distance to see if it's within the epsilon value
            if(distance <= self.epsilon and distance != 0):
                neighbors.append(i)

        return neighbors

    #Cluster points through DBSCAN and return which point belongs to which cluster, returns 1 if successful
    def cluster(self, data):
        #Creates an array of labels starting at 0 indicating if a point is in a cluster
        self.labels = np.zeros(data.shape[0])
        cluster = 0

        for labelIndex, dataPoint in enumerate(data):
            if self.labels[labelIndex] != 0:
                #point is already assigned to the cluster
                continue
            
            #Gets the neighbors for the given dataPoint
            neighbors = self.getNeighbors(self, dataPoint, data)

            #Checks to see if the given neighbor is a core point
            if len(neighbors) >= self.minPoints:
                cluster += 1
                self.labels[labelIndex] = cluster

                #Goes through the neighbors
                for i in neighbors:
                    #If a neighor already is in a cluster, skip it
                    if self.labels[i] != 0:
                        continue

                    self.labels[i] = cluster

                    #Include border points also into the cluster
                    borderNeighbors = self.getNeighbors(self, data[i], data)
                    #checks to see if the neighbor's neighbor itself is a core point
                    if(len(borderNeighbors) >= self.minPoints):
                        #Adds the border Neighbor to the neighbors to get it's neighbors
                        neighbors += borderNeighbors

        return 1
    
    #A functions that just tells how many points are clustered versus how many aren't
    def getAmountInCluster(self):
        notClustered = Clustered = 0
        for label in self.labels:
            if label >= 1:
                Clustered += 1
            else:
                notClustered += 1
        
        return [Clustered, notClustered]
    

def main():

    training = np.load(r".\KDD99\training_normal.npy")
    testingNormal = np.load(r".\KDD99\testing_normal.npy")
    testingAttack = np.load(r".\KDD99\testing_attack.npy")

    dataGraph(training, testingNormal)

    testingNormalPCA = PCAcalcs(training, testingNormal)
    testingAttackPCA = PCAcalcs(training, testingAttack)

    kmeans = KMeans(k=10)
    labels = kmeans.fit(testingNormalPCA, 200)

    plt.scatter(testingNormalPCA[:, 0], testingNormalPCA[:, 1], c=labels)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c="black", marker="*", s=200)
    plt.show()

    
    kmeans = KMeans(k=10)
    labels = kmeans.fit(testingAttackPCA, 200)

    plt.scatter(testingAttackPCA[:, 0], testingAttackPCA[:, 1], c=labels)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c="black", marker="*", s=200)
    plt.show()


    '''
    silhouette_scores = []
    for k in range(3, 15, 2):
        test_means = KMeans(k, metrics=False)
        labels = test_means.fit(testingNormalPCA)
        silhouette_scores.append(silhouette_score(testingNormalPCA, labels, metric='euclidean'))
    
    plt.plot(range(3, 15, 2), silhouette_scores)
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.show()
    '''

    DBs = DBSCAN(epsilon=.7, minPoints=6)
    DBs.cluster(testingNormalPCA)
    TruePosOrFalseNeg = DBs.getAmountInCluster()
    TP = TruePosOrFalseNeg[0] 
    FN = TruePosOrFalseNeg[1]
    plt.scatter(testingNormalPCA[:,0], testingNormalPCA[:,1], marker='.', c=DBs.getLabels(), cmap='rainbow')
    plt.show()

    DBs.cluster(testingAttackPCA)
    holder = DBs.getAmountInCluster()
    FP = holder[0]
    TN = holder[1]

    tpr = TP / (FN + TP)
    fpr = FP / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1_score = (2*TP) / (2*TP + FP + FN)
    print("DBScan Metrics")
    print("TPR = {tpr}", tpr)
    print("FPR = {fpr}", fpr)
    print("Accuracy = {acc}", accuracy)
    print("F-1 Score = {f1}", f1_score)
    


main()

