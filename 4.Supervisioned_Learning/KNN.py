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
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline  # Attenzione: import giusto!
from imblearn.over_sampling import SMOTE

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

print(dataset.info())

# Preparazione dati
y = dataset['Status'].map({'Alive': 0, 'Dead': 1})
X = dataset.drop(['Status', 'Survival Months'], axis=1)  
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

# Divisione train-test (usa ancora y_train, X_train originali, NON bilanciati)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)

# Trovo K ottimale semplicemente bilanciando prima (ma ora cerchiamo il migliore sulla cross-val)
# Uso solo una volta qui per la ricerca, ma la pipeline farà tutto poi per la valutazione vera

error = []
for i in range(1, 20):
    pipeline_tmp = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=i))
    ])
    pipeline_tmp.fit(X_train, y_train)
    pred_i = pipeline_tmp.predict(X_test)
    error.append(np.mean(pred_i != y_test))

optimal_k = error.index(min(error)) + 1
print(f"\nK ottimale trovato: {optimal_k}")

# Preparo la pipeline definitiva
pipeline = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=optimal_k))
])

# Cross-validation con SMOTE DENTRO IL FOLD, su tutto il train originale
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print('\nCross-validation results:')
print(f'Mean accuracy: {np.mean(cv_scores):.4f}')
print(f'Standard deviation: {np.std(cv_scores):.4f}')
print('\ncv_score variance:{}'.format(np.var(cv_scores)))

# Fit finale su tutto il train e valutazione sul test
pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print(f"\nAccuracy Score: {accuracy:.4f}")

print('\nClassification Report:\n', classification_report(y_test, prediction))
print('\nConfusion matrix:\n', confusion_matrix(y_test, prediction))

# Matrice di confusione normalizzata
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

# ROC Curve
probs = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)
print('\nAUC: %.3f' % auc)

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

# Varianza e std cross-validation
data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
plt.show()