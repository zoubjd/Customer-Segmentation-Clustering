import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


customers_data = pd.read_csv('Mall_Customers.csv')
copy_customers_data = customers_data.copy()
print(customers_data.columns)

scaler = StandardScaler()
customers_data[['AnnualIncome', 'SpendingScore']] = scaler.fit_transform(customers_data[['AnnualIncome', 'SpendingScore']])


plt.scatter(copy_customers_data["AnnualIncome"], copy_customers_data["SpendingScore"], color="blue", alpha=0.5)  # scatter plot of hours vs scores
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Annual Income vs Spending Score")
plt.show()

cluster = 0
num_clusters = 0

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(customers_data[['AnnualIncome', 'SpendingScore']])
    score = silhouette_score(customers_data[['AnnualIncome', 'SpendingScore']], labels)
    if score > cluster:
        cluster = score
        num_clusters = k
    print(f"k={k}, silhouette score={score:.3f}")

print(f"The highest value is: {cluster}", num_clusters)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)

kmeans.fit(customers_data[['AnnualIncome', 'SpendingScore']])

plt.scatter(copy_customers_data["AnnualIncome"], copy_customers_data["SpendingScore"], c=kmeans.labels_)  # scatter plot of hours vs scores
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Kmeans Clustering")
plt.show()

copy_customers_data["KMeansCluster"] = kmeans.labels_

kmeans_summary = copy_customers_data.groupby("KMeansCluster")[["AnnualIncome", "SpendingScore"]].mean()
print("KMeans cluster averages:")
print(kmeans_summary)

min_samples = num_clusters
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(customers_data[['AnnualIncome','SpendingScore']])
distances, indices = neighbors_fit.kneighbors(customers_data[['AnnualIncome','SpendingScore']])

distances = np.sort(distances[:, min_samples-1])  
plt.plot(distances)
plt.ylabel("k-distance")
plt.xlabel("Points sorted by distance")
plt.title("k-distance graph for DBSCAN")
plt.show()

n_clusters = 0
best_eps = 0
best_score = 0

#min samples is usually 2 * number of features
for eps in [0.3, 0.35, 0.4, 0.45, 0.5]:
    db = DBSCAN(eps=eps, min_samples=5).fit(customers_data[['AnnualIncome', 'SpendingScore']])
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        score = silhouette_score(customers_data[['AnnualIncome', 'SpendingScore']], labels)
    else:
        score = -1
    if score > best_score:
        best_score = score
        best_eps = eps
        best_n_clusters = n_clusters
    print(f"eps={eps}, clusters={n_clusters}, silhouette={score:.3f}")

dbscan = DBSCAN(eps=best_eps, min_samples=5).fit(customers_data[['AnnualIncome', 'SpendingScore']])

plt.scatter(copy_customers_data["AnnualIncome"], copy_customers_data["SpendingScore"], c=dbscan.labels_)  # scatter plot of hours vs scores
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("DBSCAN Clustering")
plt.show()

copy_customers_data["DBSCANClustering"] = dbscan.labels_

kmeans_summary = copy_customers_data.groupby("DBSCANClustering")[["AnnualIncome", "SpendingScore"]].mean()
print("DBscan cluster averages:")
print(kmeans_summary)
