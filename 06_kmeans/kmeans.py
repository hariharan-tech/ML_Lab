import ctypes

# For importing the struct defined in dist_func.so
class Point2D(ctypes.Structure):
    _fields_ = [("x",ctypes.c_float),("y",ctypes.c_float)]

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

# print(p.x,p.y)