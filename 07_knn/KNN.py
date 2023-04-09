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


# KNN class
class KNN:
    dist_types_built = ["Euclidean", "Manhattan", "Minkowski", "Supremum"]
    points = list()
    clusters = list()

    def __init__(self, pd_df, K_value, dist_type):
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
        # Dynamic cluster value selection has to be done
        for _, row in pd_df.iterrows():
            self.points.append(Point2D(row["X1"], row["X2"]))
            self.clusters.append(row["cluster"])
        self.cluster_vals = pd_df["cluster"].unique()

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
