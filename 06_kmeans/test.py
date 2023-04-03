import pandas as pd
import KMeans

# Testing
df = pd.read_csv("kmeans_dataframe.csv",delimiter=",") # Read the csv file using pandas
kmeans = KMeans.KMeans(df,3,"Euclidean") # KMeans constructor initialization


print(kmeans.set_centroids(cent_list=[4,10,8])) # Set centroids using the index provided

print(kmeans.kmeans_iter())


# clusters, new_cent = kmeans._kmeans_cluster()

''' Testing the args and kwargs'''

# for i in range(kmeans.k_value):
#     print(kmeans.centroids[i].return_pts())


# print(kmeans.set_centroids()) # Or randomly generate centroid index
# for i in range(kmeans.k_value):
#     print(kmeans.centroids[i].return_pts())

# # Testing the distance function

# for i in range(len(kmeans.points)):
#     print(kmeans.get_dist(kmeans.centroids[0],kmeans.points[i]))
#     print()