import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Wczytanie pliku CSV
data = pd.read_csv('data/Ads-data.csv')

# Wybór odpowiednich kolumn z danymi do klasteryzacji
columns = ['Age', 'EstimatedSalary']

# Wykonanie klasteryzacji
kmeans = KMeans(n_clusters=10)  # Liczba klastrów do wyboru
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

plt.title('Clusters of Social Network Ads (K-means Clustering Model)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()