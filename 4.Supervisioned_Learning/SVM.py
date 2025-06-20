import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from inspect import signature
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn import svm

# 1. Caricamento del dataset
def load_dataset():
    paths = [
        "../2.Ontologia/Breast_Cancer.csv",
        "2.Ontologia/Breast_Cancer.csv",
        "../../2.Ontologia/Breast_Cancer.csv",
        "Breast_Cancer.csv"
    ]

    for path in paths:
        try:
            dataset = pd.read_csv(path)
            # Creazione target
            if 'Status' in dataset.columns:
                dataset['diagnosis'] = dataset['Status'].map({'Alive': 0, 'Dead': 1})
                print("Colonna target creata da 'Status'")
            elif 'diagnosis' in dataset.columns:
                dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})
            else:
                raise ValueError("Nessuna colonna target trovata")
            return dataset
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Nessun file trovato nei percorsi specificati")


dataset = load_dataset()

# 2. Analisi Esplorativa
print("\n=== ANALISI ESPLORATIVA ===")
print("\nDistribuzione delle classi:")
print(dataset['diagnosis'].value_counts(normalize=True))

# 3. Preprocessing
# Separazione features e target
y = dataset['diagnosis']
X = dataset.drop(['diagnosis', 'Status'], axis=1, errors='ignore')

# Identificazione colonne numeriche e categoriche
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print("\nColonne numeriche:", numerical_cols)
print("Colonne categoriche:", categorical_cols)

# Pipeline di preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# 4. Divisione del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

# 5. Definizione del modello e ricerca iperparametri
# Crea la pipeline con SMOTE dopo il preprocessor
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', svm.SVC(kernel='rbf', probability=True, class_weight='balanced'))
])

param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [1e-4, 1e-3, 0.01, 0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 6. Miglior modello
best_model = grid_search.best_estimator_
print("\n=== MIGLIORI IPERPARAMETRI ===")
print(grid_search.best_params_)

# 7. Valutazione
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]

print("\n=== VALUTAZIONE ===")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Matrice di confusione normalizzata in percentuale
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix_percent,
    annot=True, fmt='.2f', cmap='Blues',
    xticklabels=['Pred Alive', 'Pred Dead'],
    yticklabels=['Actual Alive', 'Actual Dead']
)
plt.title("Matrice di Confusione Normalizzata (%)")
plt.ylabel("Valore Reale")
plt.xlabel("Predizione")
plt.show()

# 8. Curve di valutazione
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# Precision-Recall Curve
average_precision = average_precision_score(y_test, y_probs)
precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
plt.show()

# 9. Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1')
print("\nCross-validation F1 scores:", cv_scores)
print("Media F1:", np.mean(cv_scores))
print("Deviazione standard:", np.std(cv_scores))

# 10. Grafico varianza e deviazione standard
data = {
    'Variance': np.var(cv_scores),
    'Std Dev': np.std(cv_scores)
}
names = list(data.keys())
values = list(data.values())

plt.figure(figsize=(6, 4))
plt.bar(names, values, color=['skyblue', 'salmon'])
plt.title("Dispersione dei punteggi di Cross-Validation")
plt.ylabel("Valore")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()