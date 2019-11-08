# K-means
Python Implementation of the [K-means Clustering](https://en.wikipedia.org/wiki/K-means_clustering) algorithm.

Basic steps:

1.  Select K many random points, same dimensions as a point of one's training data, to act as centroids for one's clusters.
    * To determine the best number of K to use for a data set, run multiple trials to completion with different Ks. Calculate the WCSS for each K. Choose the K that with maximum utility; lowest K for lowest WCSS. [Elbow method for more info.](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set#The_elbow_method) ![Elbow method determination of K. Source: https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203](https://miro.medium.com/max/2190/1*vLTnh9xdgHvyC8WDNwcQQw.png) 
2.  
