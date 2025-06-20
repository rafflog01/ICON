import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from inspect import signature
import warnings

warnings.filterwarnings('ignore')


# 1. Caricamento del dataset
def load_dataset():
    paths = [
        "2.Ontologia/Breast_Cancer.csv",
        "../2.Ontologia/Breast_Cancer.csv",
        "Breast_Cancer.csv",
        "../../2.Ontologia/Breast_Cancer.csv"
    ]

    for path in paths:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Nessun file trovato nei percorsi specificati")


dataset = load_dataset()

# 2. Analisi Esplorativa dei Dati
print("\n=== ANALISI ESPLORATIVA ===\n")

# Distribuzione delle classi
print("Distribuzione della variabile target (Status):")
print(dataset['Status'].value_counts(normalize=True))

# Creazione della variabile target
dataset['target'] = dataset['Status'].apply(lambda x: 1 if x == 'Dead' else 0)

# Identificazione delle colonne numeriche e categoriche
numerical_cols = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()

# Rimuovi colonne non utili
numerical_cols = [col for col in numerical_cols if col not in ['target']]
categorical_cols = [col for col in categorical_cols if col not in ['Status']]

print("\nColonne numeriche:", numerical_cols)
print("Colonne categoriche:", categorical_cols)

# 3. Preprocessing
# Divisione in features e target
X = dataset.drop(['target', 'Status'], axis=1)
y = dataset['target']

# Preprocessing per colonne categoriche e numeriche
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# 4. Divisione del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 5. Pipeline del modello
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Pipeline con preprocessor + SMOTE + RandomForest
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Ottimizzazione iperparametri con GridSearchCV sulla pipeline con SMOTE
param_grid = {
    'classifier__max_depth': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\n=== MIGLIORI IPERPARAMETRI ===")
print(grid_search.best_params_)

# Resto della valutazione esattamente come prima...
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# 7. Miglior modello
# best_model = grid_search.best_estimator_
# print("\n=== MIGLIORI IPERPARAMETRI ===")
# print(grid_search.best_params_)

# 8. Valutazione del modello
# # Previsioni
# y_pred = best_model.predict(X_test)
# y_probs = best_model.predict_proba(X_test)[:, 1]

# Metriche
# print("\n=== VALUTAZIONE DEL MODELLO ===")
# print("\nAccuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification report:\n", classification_report(y_test, y_pred))

# Matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Alive', 'Dead'], yticklabels=['Alive', 'Dead'])
plt.title('Matrice di confusione normalizzata')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 9. Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1')
print("\nCross-validation scores:", cv_scores)
print("Media F1-score:", np.mean(cv_scores))
print("Deviazione standard:", np.std(cv_scores))

# 10. Curve ROC e Precision-Recall
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc_score))
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
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()

# Salvataggio dei risultati
results = {
    'best_params': grid_search.best_params_,
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'auc': auc_score,
    'average_precision': average_precision,
    'cv_mean_f1': np.mean(cv_scores),
    'cv_std_f1': np.std(cv_scores),
}

print("\nRisultati completi salvati nella variabile 'results'")

# Grafico varianza F1-score
data = {
    'Deviazione Standard': np.std(cv_scores),
    'Media F1': np.mean(cv_scores)
}
plt.bar(data.keys(), data.values(), color=['gray', 'green'])
plt.title('Performance della Random Forest')
plt.ylabel('Valore')
plt.show()