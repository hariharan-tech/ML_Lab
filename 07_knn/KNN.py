import ctypes
import pandas as pd
from random import randint
import numpy as np

# Configurations
# For importing the struct defined in dist_func.so
class Point2D(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

    def return_pts(self):
        """
        Return the defined point as a string to view the defined points
        """
        return f"({self.x:.2f},{self.y:.2f})"


# Load the source object file
lib = ctypes.CDLL("./dist_func.so")


def minkowski2d(point1, point2, p):
    return abs(((point1.x - point2.x) ** (p) + (point1.y - point2.y) ** (p)) ** (1 / p))


# The functions defined are mapped
funcs_defined = {
    "Euclidean": lib.euclidean2d,
    "Manhattan": lib.manhattan2d,
    "Minkowski": minkowski2d,
    "Supremum": lib.supremum2d,
    "Cosine Similarity": lib.cossim2d,
}

# For 2D K-Means (input and output argtypes) - for all functions
for i in funcs_defined:
    # In case of Minkowski distance, we need to pass another argument as well
    if i != "Minkowski":
        funcs_defined[i].argtypes = [Point2D] * 2
    else:
        funcs_defined[i].argtypes = [Point2D, Point2D, ctypes.c_float]
    # Return Type is always Float
    funcs_defined[i].restype = ctypes.c_float


class KMeans:
    dist_types_built = ["Euclidean","Manhattan","Minkowski","Supremum"]
    points = list()
    cluster = list()
    centroids = list()
    cent_list = list()
    dist_type = ""
    p = 2

    def __init__(self,pd_df,K_value,dist_type):
        '''
            KMeans Class Constructor accepts:
                - A pandas dataframe to create the points
                - k_value to indicate the number of clusters
                - dist_type to indicate the type of distance calculation
                    - Built-in types are ["Euclidean","Manhattan","Minkowski","Supremum","Cosine Similarity"]
            
            NOTE: 
                - Minkowski distance is calculated by a Python function
                - Cosine similarity doesn't work

        '''

        self.points = list()
        if dist_type not in funcs_defined or dist_type=="Cosine Similarity":
            raise Exception("Undefined distance function was specified")
        self.dist_type = dist_type
        self.k_value = K_value
        for _, row in pd_df.iterrows():
            self.points.append(Point2D(row["X1"],row["X2"]))
            # print(self.points[-1].x,self.points[-1].y)
            # print(self.points[-1].return_pts())
            # print()

    def set_centroids(self,**kwargs):
        '''
            INPUT:
                - 2 Ways of providing the centroid points:
                    - Just calling the function would randomly generate take in centroid points
                    - Instead you can pass in the centroid as index from the dataset, it will 
                        assign those points as centroid
            
            OUTPUT:
                - A list of the index of Point2D values selected from the dataframe given
        '''

        self.centroids = list()
        if len(kwargs.items()):
            cent_list = kwargs["cent_list"]
            if len(cent_list) != self.k_value:
                raise Exception("Given list of centroids doesn't match with the K values")
            for i in cent_list:
                self.centroids.append(self.points[i])
        
        else:
            cent_list = list()
            while len(cent_list)!=self.k_value:
                cent_list.append(randint(0,len(self.points)-1))
                cent_list = list(set(cent_list))
            cent_list.sort()
            for i in cent_list:
                self.centroids.append(self.points[i])
        
        self.cent_list = cent_list
        return cent_list

    def set_p(self,p):
        if self.dist_type == "Minkowski":
            self.p = p

    def get_dist(self,point1,point2,**kwargs):
        '''
            Calculates the distance (of defined type) between 2 points and returns it
            In case of Minkowski distance, you have to pass a keyword argument name p
        '''
        if self.dist_type!="Minkowski":
            return funcs_defined[self.dist_type](point1,point2)
        elif self.dist_type=="Minkowski" and len(kwargs)!=0:
            # Getting minkowski p value
            return funcs_defined["Minkowski"](point1,point2,kwargs["p"])
        else:
            raise Exception("Use the function properly!")


    def _kmeans_cluster(self):
        '''
            INPUT: None (performs 1 iteration of K-Means on the given data)

            OUTPUT:
                - A tuple, whose first element is a list that contains the cluster number of each point in data
                 and the second element is the list Point2D objects containing info. of new centroid points
        '''
        self.cluster = [0]*len(self.points)

        for i in range(len(self.points)):
            distances = list()
            for j in range(len(self.centroids)):
                distances.append(self.get_dist(self.points[i],self.centroids[j],p=self.p))
            self.cluster[i] = distances.index(min(distances))
        
        # Finding new centroids
        count = [0]*len(self.cent_list)
        for i in range(len(self.cent_list)):
            count[i] = self.cluster.count(i)


        # Making a new temporary centroid element
        temp_cent = list()
        for i in range(len(self.centroids)):
            temp_cent.append(Point2D(0.0,0.0))

        for i in range(len(self.cluster)):
            temp_cent[self.cluster[i]].x += (self.points[i].x/count[self.cluster[i]])
            temp_cent[self.cluster[i]].y += (self.points[i].y/count[self.cluster[i]])
            # print(self.cluster[i],self.points[i].x,temp_cent[0].x)
        
        # for i in temp_cent:
        #     print(i.x,i.y)

        self.centroids = temp_cent
        del temp_cent, count

        return (self.cluster,self.centroids)

    def kmeans_iter(self):
        '''
            INPUT: None (Applies Kmeans_clustering over iteration on the provided data of a particular kmeans object)
            
            OUTPUT: 
                - Performs K-means (by internally calling _kmeans_cluster) for
                  a number of iterations until the data is clustered in the same class.
                - Returns the clustering data once the data settles

        '''
        old_cluster, _ = self._kmeans_cluster()
        while True:
            new_cluster, _ = self._kmeans_cluster()
            if new_cluster==old_cluster:
                break
            old_cluster = new_cluster
        del old_cluster
        return new_cluster


# KNN class
class KNN:
    dist_types_built = ["Euclidean", "Manhattan", "Minkowski", "Supremum"]
    points = list()
    clusters = list()

    def __init__(self, pd_df, K_value, dist_type,K_means_val):
        """
        KNN Class Constructor accepts:
            - A pandas dataframe to create the points
            - k_value to indicate the number of clusters
            - dist_type to indicate the type of distance calculation
                - Built-in types are ["Euclidean","Manhattan","Minkowski","Supremum","Cosine Similarity"]

        NOTE:
            - Minkowski distance is calculated by a Python function
            - Cosine similarity doesn't work

        """

        self.points = list()
        if dist_type not in funcs_defined or dist_type == "Cosine Similarity":
            raise Exception("Undefined distance function was specified")
        self.dist_type = dist_type
        self.k_value = K_value
        # Dynamic cluster value selection has to be done - using kmeans
        # self.clusters = list()
        for _, row in pd_df.iterrows():
            self.points.append(Point2D(row["X1"], row["X2"]))
            # self.clusters.append(row["cluster"])
        kmeans = KMeans(pd_df,K_means_val,dist_type)
        self.centroids = kmeans.set_centroids()
        self.clusters = kmeans.kmeans_iter()
        # self.cluster_vals = np.array(self.clusters).unique()
        self.cluster_vals = list(set(self.clusters))

    def get_dist(self, point1, point2, **kwargs):
        """
        Calculates the distance (of defined type) between 2 points and returns it
        In case of Minkowski distance, you have to pass a keyword argument name p
        """
        if self.dist_type != "Minkowski":
            return funcs_defined[self.dist_type](point1, point2)
        elif self.dist_type == "Minkowski" and len(kwargs) != 0:
            # Getting minkowski p value
            return funcs_defined["Minkowski"](point1, point2, kwargs["p"])
        else:
            raise Exception("Use the function properly!")

    def knn_cluster(self, point):
        dist = list()
        indexes = list()
        for i in range(len(self.points)):
            dist.append(self.get_dist(self.points[i], point))
        indexes = np.argsort(dist)
        dist.sort()
        neighbours = dict.fromkeys(self.cluster_vals, 0)
        for i in range(0, self.k_value, 1):
            neighbours[self.cluster_vals[self.clusters[indexes[i]]]] += 1
        print(neighbours)
        neigh_vals = list(neighbours.values())
        max_closest = neigh_vals.index(max(neigh_vals))
        return list(neighbours.keys())[max_closest]

    # def knn_regress(self,point):
    #     dist = list()
    #     indexes = list()
    #     for i in range(len(self.points)):
    #         dist.append(self.get_dist(self.points[i], point))
    #     indexes = np.argsort(dist)
    #     dist.sort()



# knn.centroids => [41,102,141]