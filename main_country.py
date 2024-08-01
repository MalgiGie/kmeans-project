import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Wczytanie pliku CSV
data = pd.read_csv('data/Country-data.csv')

# Wybór odpowiednich kolumn z danymi do klasteryzacji
columns = ['imports', 'exports']

# Wykonanie klasteryzacji
kmeans = KMeans(n_clusters=3)  # Liczba klastrów do wyboru
kmeans.fit(data[columns])

# Przypisanie etykiet klastrów do danych
data['Cluster'] = kmeans.labels_

# Tworzenie wykresu
X = data[columns].values
labels = kmeans.labels_
clusters = set(labels)

plt.figure(figsize=(8, 6))

for cluster in clusters:
    plt.scatter(X[labels == cluster, 1], X[labels == cluster, 0], s=100, label=f'Cluster {cluster+1}')  # Odwrócone osie

plt.title('Clusters of Country Data (K-means Clustering Model)')
plt.xlabel('Export')
plt.ylabel('Import')
plt.show()
