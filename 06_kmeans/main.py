import pandas as pd
import KMeans

# Testing
df = pd.read_csv("kmeans_dataframe.csv",delimiter=",")
kmeans = KMeans.KMeans(df,3,"Euclidean")
kmeans.set_centroids([1,2,3])

for i in range(kmeans.k_value):
    print(kmeans.centroids[i].return_pts())

for i in range(len(kmeans.points)):
    print(kmeans.get_dist(kmeans.centroids[0],kmeans.points[i]))