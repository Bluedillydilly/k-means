"""

"""
import numpy as np
from copy import deepcopy

THRESHOLD = 0

def kmeans(training, TUNE_K = 0, PRINT = 0):
    """

    1. Choose K. Choose location of centroid.
    2. Assign each point of input to K.
    3. Update centroid with average of points within centroid range.
    4. repear 2 and 3 until no input point is reassigned.
    """
    dimensions = training[0].shape
    K_num = 2 # number of centroids
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
            kMean = np.mean(kPoints)
            centroids[k] = kMean
    if PRINT:
        # OUTPUT CENTROIDS
        print("Centroids",centroids)
        print("Training data",training)
    return (training,centroids)
            

def dists(point, centroids):
    return [dist(point,c) for c in centroids]

def dist(point, centroid):
    """
    Euclidean distance of point from centroid.
    """
    return np.linalg.norm(point-centroid)



if __name__ == "__main__":
    test = np.array([[1,1],[2,2],[9,10],[11,10]], dtype=float)
    maxValue = np.max(test)
    kmeans(test, PRINT=1)