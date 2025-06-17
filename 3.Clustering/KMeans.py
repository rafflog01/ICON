import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Caricamento del dataset
try:
    dataset = pd.read_csv("Breast_Cancer.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("../2.Ontologia/Breast_Cancer.csv")
    except FileNotFoundError:
        dataset = pd.read_csv("2.Ontologia/Breast_Cancer.csv")

# Verifica se la colonna 'diagnosis' esiste e, se non esiste, usa 'Status' come colonna target
if 'diagnosis' not in dataset.columns:
    if 'Status' in dataset.columns:
        dataset['diagnosis'] = dataset['Status'].map({'Alive': 0, 'Dead': 1})
    else:
        print("Attenzione: Colonna 'diagnosis' non trovata. Creata colonna vuota.")
        dataset['diagnosis'] = 0
else:
    dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})

dataset.drop(columns=dataset.columns[dataset.columns.str.contains('unnamed', case=False)], inplace=True)

# Rimuovere le colonne per il clustering
columns_to_drop = ['diagnosis']
if 'id' in dataset.columns:
    columns_to_drop.append('id')

X = dataset.drop(columns_to_drop, axis=1)

# Selezione solo delle colonne numeriche per evitare errori di conversione
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
X = X[numeric_columns]

# Standardizzazione dei dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calcolo WCSS per diversi valori di K
wcss = []
silhouette_scores = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    else:
        silhouette_scores.append(0)

# Elbow plot
plt.plot(k_range, wcss, 'bx-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Addestramento del modello finale con K=2 
kmeans_final = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans_final.fit(X_scaled)

# Stampa metriche
print(f"\n-WCSS: {kmeans_final.inertia_:.2f}")
print(f"-Silhouette Score: {silhouette_score(X_scaled, kmeans_final.labels_):.4f}")

# Aggiunta etichetta cluster al dataset originale
dataset['cluster'] = kmeans_final.labels_

# Riordina le colonne del DataFrame
columns_order = list(dataset.columns)
if 'diagnosis' in columns_order:
    columns_order.remove('diagnosis')
    columns_order.append('diagnosis')
if 'cluster' in columns_order:
    columns_order.remove('cluster')
    columns_order.insert(-1, 'cluster')
dataset_reordered = dataset[columns_order]

dataset_reordered.to_csv('Breast_Cancer_KMeans-clusters.csv', index=False)