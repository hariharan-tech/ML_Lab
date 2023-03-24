import ctypes
import pandas as pd

# Configurations
# For importing the struct defined in dist_func.so
class Point2D(ctypes.Structure):
    _fields_ = [("x",ctypes.c_float),("y",ctypes.c_float)]

    def return_pts(self):
        '''
            Return the defined point as a string to view the defined points
        '''
        return f"({self.x:.2f},{self.y:.2f})"

# Load the source object file
lib = ctypes.CDLL("./dist_func.so")

# The functions defined are mapped
funcs_defined = {
    "Euclidean":lib.euclidean2d,
    "Manhattan":lib.manhattan2d,
    "Minkowski":lib.minkowski2d,
    "Supremum":lib.supremum2d,
    "Cosine Similarity":lib.cossim2d
    }

# For 2D K-Means (input and output argtypes) - for all functions
for i in funcs_defined:
    # In case of Minkowski distance, we need to pass another argument as well
    if i != "Minkowski":
        funcs_defined[i].argtypes = [Point2D]*2
    else:
        funcs_defined[i].argtypes = [Point2D,Point2D,ctypes.c_float]
    # Return Type is always Float
    funcs_defined[i].restype = ctypes.c_float


# KMeans Class
class KMeans:
    points = list()
    centroids = list()
    dist_type = ""

    def __init__(self,pd_df,K_value,dist_type):
        '''
            KMeans Class Constructor accepts:
                - A pandas dataframe to create the points
                - k_value to indicate the number of clusters
                - dist_type to indicate the type of distance calculation
                    - Built-in types are ["Euclidean","Manhattan","Minkowski","Supremum","Cosine Similarity"]

        '''

        if dist_type not in funcs_defined:
            raise Exception("Undefined distance function was specified")
        self.dist_type = dist_type
        self.k_value = K_value
        for _, row in pd_df.iterrows():
            self.points.append(Point2D(row["X1"],row["X2"]))
            # print(self.points[-1].x,self.points[-1].y)
            # print(self.points[-1].return_pts())
            # print()
    
    def set_centroids(self,cent_list):
        '''
            Set the K - centroids from a given index of the input data from cent_list (input)
        '''

        if len(cent_list) != self.k_value:
            raise Exception("Given list of centroids doesn't match with the K values")
        for i in cent_list:
            self.centroids.append(self.points[i])

    def get_dist(self,point1,point2):
        '''
            Calculates the distance (of defined type) between 2 points and returns it
        '''

        return funcs_defined[self.dist_type](point1,point2)


'''

# Creating nodes
p1 = Point2D(3,0)
p2 = Point2D(0,4)

# These points are accessible in python as well!
print(f"Point 1: ({p1.x},{p1.y})\nPoint 2: ({p2.x},{p2.y})")



# Testing distance values
dist = lib.euclidean2d(p1,p2)
print(dist)
dist = lib.manhattan2d(p1,p2)
print(dist)
dist = lib.minkowski2d(p1,p2,2)
print(dist)
dist = lib.supremum2d(p1,p2)
print(dist)
dist = lib.cossim2d(p1,p2)
# If dist==-2 (divide by zero error case, exit the program by raising an exception)
if dist==-2:
    raise Exception("Divide by zero error")
print(dist)

# print(p.x,p.y) '''