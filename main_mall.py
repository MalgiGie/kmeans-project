import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Wczytanie pliku CSV
data = pd.read_csv('data/Mall-data.csv')

# Wybór odpowiednich kolumn z danymi do klasteryzacji
columns = ['Annual Income (k$)', 'Spending Score (1-100)']

# Wykonanie klasteryzacji
kmeans = KMeans(n_clusters=5)  # Liczba klastrów do wyboru
kmeans.fit(data[columns])

# Przypisanie etykiet klastrów do danych
data['Cluster'] = kmeans.labels_

# Tworzenie wykresu
X = data[columns].values
labels = kmeans.labels_
clusters = set(labels)

plt.figure(figsize=(8, 6))

for cluster in clusters:
    plt.scatter(X[labels == cluster, 0], X[labels == cluster, 1], s=100, label=f'Cluster {cluster+1}')

plt.title('Clusters of Mall Customers (K-means Clustering Model)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()