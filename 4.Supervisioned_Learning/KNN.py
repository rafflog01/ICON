
import numpy as np
import pandas as pd
import seaborn as sn
import os
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from inspect import signature
from sklearn.metrics import average_precision_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Stampa la directory corrente per debug
print(f"Directory di lavoro corrente: {os.getcwd()}")

# Caricamento del dataset
try:
    dataset = pd.read_csv("Breast_Cancer.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("../2.Ontologia/Breast_Cancer.csv")
    except FileNotFoundError:
        try:
            dataset = pd.read_csv("2.Ontologia/Breast_Cancer.csv")
        except FileNotFoundError:
            dataset = pd.read_csv("../../2.Ontologia/Breast_Cancer.csv")

# Preprocessing
# Verifica se la colonna 'diagnosis' esiste e, se non esiste, usa 'Status' come colonna target
if 'diagnosis' not in dataset.columns:
    if 'Status' in dataset.columns:
        dataset['diagnosis'] = dataset['Status'].map({'Alive': 0, 'Dead': 1})
        print("Colonna 'diagnosis' creata dalla colonna 'Status'")
    else:
        print("Attenzione: Colonna 'diagnosis' non trovata e colonna 'Status' non disponibile.")
        dataset['diagnosis'] = 0
else:
    dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})

print(dataset.info())

# Preparazione dati
y = dataset['diagnosis']
X = dataset.drop(['diagnosis'], axis=1)
# Rimuovi 'id' se esiste
if 'id' in X.columns:
    X = X.drop(['id'], axis=1)
X.drop(X.columns[X.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

# Identifica colonne categoriche e numeriche
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

print(f"Colonne categoriche trovate: {list(categorical_cols)}")
print(f"Colonne numeriche: {list(numerical_cols)}")

# Utilizziamo solo le colonne numeriche per l'analisi KNN
X = X[numerical_cols]

# Divisione train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)

# Standardizzazione
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ricerca del K ottimale
error = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled, y_train)
    pred_i = knn.predict(X_test_scaled)
    error.append(np.mean(pred_i != y_test))

# Visualizzazione error rate
plt.figure(figsize=(10, 6))
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Determinazione K ottimale
optimal_k = error.index(min(error)) + 1
print(f"\nK ottimale trovato: {optimal_k}")

# Training con K ottimale
neigh = KNeighborsClassifier(n_neighbors=optimal_k)
neigh.fit(X_train_scaled, y_train)

# Predizioni
prediction = neigh.predict(X_test_scaled)
accuracy = accuracy_score(y_test, prediction)
print(f"Accuracy Score: {accuracy:.4f}")

# Report di classificazione
print('\nClassification Report:\n', classification_report(y_test, prediction))
print('\nConfusion matrix:\n', confusion_matrix(y_test, prediction))

# Matrice di confusione
conf_matrix = confusion_matrix(y_test, prediction)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(10, 7))
sn.heatmap(pd.DataFrame(conf_matrix_percent, index=['Benigno (0)', 'Maligno (1)'],
                       columns=['Pred Benigno (0)', 'Pred Maligno (1)']),
           annot=True, fmt='.2f', cmap='Blues')
plt.title('Matrice di Confusione Normalizzata (%)')
plt.ylabel('Valore Reale')
plt.xlabel('Predizione')
plt.show()

# Cross-validation
cv_scores = cross_val_score(neigh, X_train_scaled, y_train, cv=5)
print('\nCross-validation results:')
print(f'Mean accuracy: {np.mean(cv_scores):.4f}')
print(f'Standard deviation: {np.std(cv_scores):.4f}')
print('\ncv_score variance:{}'.format(np.var(cv_scores)))

# ROC Curve
probs = neigh.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)
print('\nAUC: %.3f' % auc)

# Calcolo della curva ROC e visualizzazione
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.xlabel('FP RATE')
pyplot.ylabel('TP RATE')
pyplot.show()

# Precision-Recall Curve
average_precision = average_precision_score(y_test, probs)
precision, recall, _ = precision_recall_curve(y_test, probs)

plt.figure(figsize=(8, 6))
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall curve (AP = {average_precision:.3f})')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# F1 Score
f1 = f1_score(y_test, prediction)
print(f'\nF1 Score: {f1:.4f}')

# Creazione di un grafico per visualizzare varianza e deviazione standard dei cv_scores
data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
plt.show()
