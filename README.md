# K-means
Python Implementation of the [K-means Clustering](https://en.wikipedia.org/wiki/K-means_clustering) algorithm.

### Usage:
To run standalone test: ```python3  K-means.py```. Run a simple test run.
Otherwise, ```import Kmeans as km ..... result = km.kmeans(training_data)``` to get a tuple of (training_data,centroids).

### TODO:
Implement the elbow method to automatically determine the best K; number of clusters.

### Notes:
If documentation is vague sorry brain potato right now and will probably neglect to change this.

### Basic Steps:

1.  Select K many random points, same dimensions as a point of one's training data, to act as centroids for one's clusters.
    * To determine the best number of K to use for a data set, run multiple trials to completion with different Ks. Calculate the WCSS for each K. Choose the K that with maximum utility; lowest K for lowest WCSS. [Elbow method for more info.](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set#The_elbow_method) ![Elbow method determination of K. Source: https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203](https://miro.medium.com/max/2190/1*vLTnh9xdgHvyC8WDNwcQQw.png)
    Source of image: https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
2.  Assign each sample in training data (input data) to the closest centroid.
    * Using euclidean distance. 
    * ||Point<sub>i</sub>-Centroid<sub>j</sub>|| = [ (Point<sub>i,1</sub> - Centroid<sub>j,1</sub>)<sup>2</sup> + (Point<sub>i,2</sub> - Centroid<sub>j,2</sub>)<sup>2</sup> + ... + (Point<sub>i,k</sub> - Centroid<sub>j,k</sub>)<sup>2</sup> + ... + (Point<sub>i,n</sub> - Centroid<sub>j,n</sub>)<sup>2</sup> ]<sup>0.5</sup>
3.  Update centroid with the average of all points currently in centroid.
4.  Repeat step **2** and **3** until no points get reassigned to a new centroid.

### WHY
K-means has seemed interesting to me for a while and wanted to use it for personal projects. I wanted to personally implement K-means to better understand it before using one from an existing library.
