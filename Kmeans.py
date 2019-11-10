"""

"""
import numpy as np
from copy import deepcopy

THRESHOLD = 0

def kmeans(training, K = 2, PRINT = 0):
    """

    1. Choose K. Choose location of centroid.
    2. Assign each point of input to K.
    3. Update centroid with average of points within centroid range.
    4. repear 2 and 3 until no input point is reassigned.

    return:
        centroids - the array representing the k many centroids.
                A data sample belongs to whatever centroid it is closest to,
                and takes that centroid's label (i.e. 0,1,2,...,k).
    """
    dimensions = training[0].shape
    K_num = K # number of centroids
    K_dim = (K_num,) + dimensions # dimensions of K
    centroids = np.random.random(K_dim) # values of K
    centroids *= np.max(training, axis=0)
    #centroids = np.array([[-10,-10],[20,20]], dtype=float)
    
    if PRINT:
        print("Training data:\n", training)
        print("Starting centroids:\n", centroids)
        print()

    # list of labels for each training
    # ie label at training_labels[i] is the centroid
    # that training[i] belongs to.
    training_labels = np.zeros(len(training)) * -1

    # centroids from previous iteration 
    OLD_centroids = np.zeros(centroids.shape)


    while dist(centroids, OLD_centroids) > THRESHOLD:    
        # ASSIGN CENTROID LABELS TO TRAINING.
        for i in range(len(training)):
            # list of distances of point training i to 
            # all the centroids
            distances = dists(training[i], centroids)
            #print("Distances", distances)
            # index of the cluster that has the lowest 
            # distance to training[i]
            cluster = np.argmin(distances)
            # assigning the label to the ith training sample
            training_labels[i] = cluster

        # assign current centroids to old before update
        OLD_centroids = deepcopy(centroids)
        # UPDATE CENTROIDS with averages
        for k in range(K_num):
            kPoints = [training[i] for i in range(len(training)) if training_labels[i] == k]
            if len(kPoints) == 0:
                continue
            kMean = np.mean(kPoints)
            centroids[k] = kMean
    if PRINT:
        # OUTPUT CENTROIDS
        print("Centroids",centroids)
        print("Training data",training)
    return (centroids, training_labels)
            

def dists(point, centroids):
    return [dist(point,c) for c in centroids]

def dist(point, centroid):
    """
    Euclidean distance of point from centroid.
    """
    return np.linalg.norm(point-centroid)


def kmeansTuned(training, PRINT=0):
    """
    Runs kmeans several times and finds the best K value (number of centroids to use).
    """
    # run kmeans with different k
    WCSSk = []
    for k in range(1,11):
        if PRINT:
            print("K: ",k)

        iterations = 15
        for i in range(iterations):
            clusterI = []
            cent, labels = kmeans(training, K=k)
            clusterI.append(calcWCSS(training, labels, cent))
        WCSSk.append(sum(clusterI)/iterations)

    if PRINT:
        print("WCCS:",WCSSk)
    candidates = [ WCSSk[i-1]-2*WCSSk[i]+WCSSk[i+1] for i in range(1,len(WCSSk)-1)]
    if PRINT:
        print("Candidates: ", candidates)
    # gets the k of the lowest WCSS
    bestK = np.argmax(np.array(candidates)) + 1 + 1
    return bestK

def calcWCSS(T, labels, C):
    """
    parameters:
        T - training data; input data.
        labels - labels for each training sample. ith label corresponds to ith training sample
        C - the k centroids

    """
    WCSSsum = []
    # calculates the WCSS per cluster.
    for i in range(len(C)):
        WCSSsum.append(sum([dist(T[j], C[i]) for j in range(len(T)) if labels[j]==i]))
    # total WCSS of all cluster
    totalWCSS = np.sum(WCSSsum)
    return totalWCSS



if __name__ == "__main__":
    test = np.array([[0,1],[3,2],[60,63],[54,57],[102,98],[99,100]], dtype=float)
    print(kmeans(test))

    bestK =kmeansTuned(test) 
    print("Best K to use:", bestK)
    print(kmeans(test, K=bestK))