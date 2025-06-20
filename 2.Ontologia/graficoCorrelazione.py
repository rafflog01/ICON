import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Caricamento dati
dataset = None
try:
    dataset = pd.read_csv("Breast_Cancer.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("../2.Ontologia/Breast_Cancer.csv")
    except FileNotFoundError:
        try:
            dataset = pd.read_csv("2.Ontologia/Breast_Cancer.csv")
        except FileNotFoundError:
            print("ERRORE: File Breast_Cancer.csv non trovato in nessuno dei percorsi.")
            exit(1)

# Pulizia nomi colonne: rimuove eventuali spazi finali
dataset.columns = dataset.columns.str.strip()

# Pulizia spazi anche nei valori di tutte le colonne categoriche principali
for col in ["Race", "Marital Status", "T Stage", "N Stage", "6th Stage", "differentiate", "A Stage", "Estrogen Status", "Progesterone Status", "Status"]:
    dataset[col] = dataset[col].astype(str).str.strip()

# Encoding delle colonne categoriche
dataset['Race'] = dataset['Race'].astype('category').cat.codes
dataset['Marital Status'] = dataset['Marital Status'].astype('category').cat.codes
dataset['T Stage'] = dataset['T Stage'].astype('category').cat.codes
dataset['N Stage'] = dataset['N Stage'].astype('category').cat.codes
dataset['6th Stage'] = dataset['6th Stage'].astype('category').cat.codes
dataset['differentiate'] = dataset['differentiate'].astype('category').cat.codes
dataset['A Stage'] = dataset['A Stage'].astype('category').cat.codes
dataset['Estrogen Status'] = dataset['Estrogen Status'].map({'Positive': 1, 'Negative': 0})
dataset['Progesterone Status'] = dataset['Progesterone Status'].map({'Positive': 1, 'Negative': 0})
dataset['Status'] = dataset['Status'].map({'Alive': 1, 'Dead': 0})

# Correggi eventuali errori nei nomi delle colonne numeriche
if 'Reginol Node Positive' in dataset.columns:
    dataset.rename(columns={'Reginol Node Positive': 'Regional Node Positive'}, inplace=True)

# Conversione forzata di 'Grade' a numerico
dataset['Grade'] = pd.to_numeric(dataset['Grade'], errors='coerce')

# Ricerca nuove colonne numeriche (ora anche 'Grade' sarà inclusa)
numerical_columns = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Genera matrice di correlazione
corr_matrix = dataset[numerical_columns].corr()

# Heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap="coolwarm", cbar=True)
plt.title("Heat Map delle Correlazioni Breast Cancer (feature codificate)")
plt.tight_layout()
plt.show()