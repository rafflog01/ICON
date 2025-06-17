import tensorflow as tf
from inspect import signature
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, \
    average_precision_score, precision_recall_curve, f1_score
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold

# Riproducibilita'
np.random.seed(42)
tf.random.set_seed(42)

# Caricamento del dataset
try:
    data = pd.read_csv("../2.Ontologia/Breast_Cancer.csv")
except FileNotFoundError:
    try:
        data = pd.read_csv("2.Ontologia/Breast_Cancer.csv")
    except FileNotFoundError:
        data = pd.read_csv("Breast_Cancer.csv")

# Creazione della colonna 'diagnosis'
data['diagnosis'] = data['Status'].apply(lambda x: 1 if x == 'Dead' else 0)

# Rimozione valori mancanti
data = data.dropna()

# Separazione feature e target
y = data['diagnosis']
X = data.drop(['diagnosis', 'Status'], axis=1)
X.drop(X.columns[X.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)

# Identificazione tipi di feature
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# One-hot encoding per le categoriche
X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True) if len(categorical_cols) > 0 else pd.DataFrame(index=X_train.index)
X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True) if len(categorical_cols) > 0 else pd.DataFrame(index=X_test.index)

# Standardizzazione numeriche
scaler = StandardScaler()
X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]), columns=numeric_cols, index=X_train.index) if len(numeric_cols) > 0 else pd.DataFrame()
X_test_num_scaled = pd.DataFrame(scaler.transform(X_test[numeric_cols]), columns=numeric_cols, index=X_test.index) if len(numeric_cols) > 0 else pd.DataFrame()

# Combinazione
X_train_scaled = pd.concat([X_train_num_scaled, X_train_cat], axis=1)
X_test_scaled = pd.concat([X_test_num_scaled, X_test_cat], axis=1)

y_train = y_train.loc[X_train_scaled.index]
y_test = y_test.loc[X_test_scaled.index]

# Costruzione modello
def create_model(input_shape=None):
    if input_shape is None:
        input_shape = X_train_scaled.shape[1]
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Pesi delle classi
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Addestramento modello principale
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model1 = create_model()
history = model1.fit(X_train_scaled, y_train, epochs=30, batch_size=64, class_weight=class_weight_dict, validation_split=0.2, callbacks=[early_stopping])

# Valutazione test
test_loss, test_accuracy = model1.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Predizioni
predictions = model1.predict(X_test_scaled)
threshold = 0.5
rounded = (predictions[:, 0] >= threshold).astype(int)

# Classification report e confusion matrix
print('\nClassification report:\n', classification_report(y_test, rounded))
conf_matrix = confusion_matrix(y_test, rounded)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
df_cm = pd.DataFrame(conf_matrix_percent, index=[i for i in "01"], columns=[i for i in "01"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')
plt.title('Matrice di confusione normalizzata (percentuali)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
X_for_cv = X_train_scaled.reset_index(drop=True)
y_for_cv = y_train.reset_index(drop=True)

for train_index, val_index in kf.split(X_for_cv, y_for_cv):
    X_train_fold = X_for_cv.iloc[train_index]
    X_val_fold = X_for_cv.iloc[val_index]
    y_train_fold = y_for_cv.iloc[train_index]
    y_val_fold = y_for_cv.iloc[val_index]

    missing_cols = set(X_train_fold.columns) - set(X_val_fold.columns)
    for col in missing_cols:
        X_val_fold[col] = 0
    X_val_fold = X_val_fold[X_train_fold.columns]

    modelK = create_model(X_train_fold.shape[1])
    modelK.fit(X_train_fold, y_train_fold, epochs=30, batch_size=64, verbose=0,
               validation_split=0.2, callbacks=[early_stopping])
    val_loss, val_accuracy = modelK.evaluate(X_val_fold, y_val_fold)
    cv_scores.append(val_accuracy)

# Statistiche CV
print('\ncv_scores mean:{}'.format(np.mean(cv_scores)))
print('\ncv_score variance:{}'.format(np.var(cv_scores)))
print('\ncv_score standard deviation:{}'.format(np.std(cv_scores)))

# ROC e AUC
probs = model1.predict(X_test_scaled)[:, 0]
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('FP RATE')
plt.ylabel('TP RATE')
plt.show()

# Precision-Recall
average_precision = average_precision_score(y_test, probs)
precision, recall, _ = precision_recall_curve(y_test, probs)
step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()

# F1-score
f1 = f1_score(y_test, rounded)
print('\nf1 score: ', f1)

# Grafico varianza e std della cross-validation
cv_stats = {'variance': np.var(cv_scores), 'standard deviation': np.std(cv_scores)}
names = list(cv_stats.keys())
values = list(cv_stats.values())
fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
plt.show()