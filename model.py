import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import joblib
import re
import logging
import warnings

# Configuration des avertissements et logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fonctions utilitaires (tirées de vos notebooks)
def nettoyer_diagnostic(diag):
    if not isinstance(diag, str):
        return diag
    nettoyé = re.sub(r'\s+', ' ', diag.strip()).upper()
    corrections = {
        'CVO OSSEUS E': 'CVO OSSEUSE', 'CVO OSSUSE': 'CVO OSSEUSE',
        'AMYDALITE': 'AMYGDALITE', 'AMYGDALITE ': 'AMYGDALITE',
        'DOULEUR ABDOMINAL ': 'DOULEUR ABDOMINALE', ' DOULEUR ABDOMINAL': 'DOULEUR ABDOMINALE',
        'RHINORHEE': 'RHINORRHEE', 'RHINORRHEE ': 'RHINORRHEE',
        'VOMISSIMENT': 'VOMISSEMENT', 'VOMISSEMENT ': 'VOMISSEMENT',
        'CEPHALEE ': 'CEPHALEE', ' CEPHALEE ': 'CEPHALEE',
        'MAL GORGE': 'MAL DE GORGE', 'MAUX GORGE': 'MAL DE GORGE'
    }
    return corrections.get(nettoyé, nettoyé)

def categoriser_diagnostic(diag):
    if not isinstance(diag, str):
        return 'Autres'
    diag = diag.upper()
    if 'CVO' in diag:
        return 'CVO'
    elif any(keyword in diag for keyword in ['AMYGDALITE', 'PHARYNGITE', 'RHINITE', 'RHINOPHARYNGITE', 'BRONCHITE', 'PNEUMONIE', 'SYNDROME INFECTIEUX', 'SYNDROME GRIPPAL']):
        return 'Infections'
    elif any(keyword in diag for keyword in ['ANEMIE', 'HEMOLYSE']):
        return 'Anémie'
    elif 'STA' in diag:
        return 'STA'
    elif 'AVCI' in diag or 'RECIDIVE AVCI' in diag:
        return 'AVC'
    else:
        return 'Autres'

# Chargement des données
try:
    df = pd.read_excel("C:/Users/DELL/Desktop/stage sante/Base_de_données_USAD_URGENCES1.xlsx", sheet_name=None)
    data = pd.concat(df.values(), ignore_index=True)
    logging.info(f"Données chargées : {data.shape[0]} lignes, {data.shape[1]} colonnes")
except Exception as e:
    logging.error(f"Erreur lors du chargement des données : {e}")
    raise

# Variables pour la prédiction 
variables_selection = [
    'Âge de début des signes (en mois)', 'NiveauUrgence', 'GR (/mm3)', 'GB (/mm3)',
    "Nbre d'hospitalisations avant 2017", 'CRP Si positive (Valeur)', 'Pâleur',
    'Âge du debut d etude en mois (en janvier 2023)', 'Souffle systolique fonctionnel',
    'VGM (fl/u3)', 'HB (g/dl)', 'Vaccin contre méningocoque', 'Nbre de GB (/mm3)',
    "% d'Hb S", 'Âge de découverte de la drépanocytose (en mois)', 'Splénomégalie',
    'Prophylaxie à la pénicilline', "Taux d'Hb (g/dL)", 'Parents Salariés',
    'PLT (/mm3)', 'Diagnostic Catégorisé', 'Prise en charge Hospitalisation',
    'Nbre de PLT (/mm3)', 'TCMH (g/dl)', 'Nbre de transfusion avant 2017',
    'Radiographie du thorax Oui ou Non', "Niveau d'instruction scolarité",
    "Nbre d'hospitalisations entre 2017 et 2023", "% d'Hb F",
    'Douleur provoquée (Os.Abdomen)', 'Mois', 'Vaccin contre pneumocoque',
    'HDJ', 'Nbre de transfusion Entre 2017 et 2023'
]

# Variables pour le clustering 
variables_clustering = [
    'Âge de début des signes (en mois)', 'Taux d\'Hb (g/dL)', '% d\'Hb F', 'Nbre de GB (/mm3)',
    'Nbre de PLT (/mm3)', 'Âge de découverte de la drépanocytose (en mois)'
]

# Prétraitement des données
# Nettoyage des diagnostics
if 'Diagnostic' in data.columns:
    data['Diagnostic'] = data['Diagnostic'].apply(nettoyer_diagnostic)
    data['Diagnostic Catégorisé'] = data['Diagnostic'].apply(categoriser_diagnostic)

# Conversion des variables Oui/Non en binaire
binary_columns = [
    'Pâleur', 'Souffle systolique fonctionnel', 'Vaccin contre méningocoque',
    'Vaccin contre pneumocoque', 'Prophylaxie à la pénicilline', 'Splénomégalie',
    'Parents Salariés', 'Prise en charge Hospitalisation', 'Radiographie du thorax Oui ou Non',
    'Douleur provoquée (Os.Abdomen)', 'HDJ'
]
for col in binary_columns:
    if col in data.columns:
        data[col] = data[col].map({'OUI': 1, 'NON': 0})

# Conversion des variables catégorielles
if 'NiveauUrgence' in data.columns:
    data['NiveauUrgence'] = data['NiveauUrgence'].map({
        'Urgence1': 1, 'Urgence2': 2, 'Urgence3': 3, 'Urgence4': 4, 'Urgence5': 5, 'Urgence6': 6
    })

if 'Niveau d\'instruction scolarité' in data.columns:
    data['Niveau d\'instruction scolarité'] = data['Niveau d\'instruction scolarité'].map({
        'Maternelle': 1, 'Elémentaire': 2, 'Secondaire': 3, 'Enseignement Supérieur': 4, 'NON': 0
    })

# One-hot encoding pour Diagnostic Catégorisé et Mois
if 'Diagnostic Catégorisé' in data.columns:
    data = pd.get_dummies(data, columns=['Diagnostic Catégorisé'], prefix='Diagnostic Catégorisé')
if 'Mois' in data.columns:
    data = pd.get_dummies(data, columns=['Mois'], prefix='Mois')

# Supprimer les lignes avec des valeurs manquantes pour les variables de prédiction
data_pred = data[variables_selection].dropna()

# Préparer les features et la cible 
if 'Evolution' in data.columns:
    X = data_pred.drop(columns=['Evolution'])
    y = data_pred['Evolution'].map({'Favorable': 0, 'Complications': 1})
else:
    logging.error("La colonne 'Evolution' est manquante. Veuillez spécifier la colonne cible.")
    raise ValueError("Colonne 'Evolution' manquante")

# Standardisation des variables quantitatives
quantitative_vars = [
    'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
    'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
    'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
    'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
    "Nbre d'hospitalisations entre 2017 et 2023",
    'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
    'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", '% d\'Hb F'
]
scaler = StandardScaler()
X[quantitative_vars] = scaler.fit_transform(X[quantitative_vars])

# Sauvegarde du scaler
joblib.dump(scaler, 'scaler.pkl')
logging.info("Scaler sauvegardé dans scaler.pkl")

# Entraînement du modèle LightGBM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data])
model.save_model('lightgbm_model.txt')
logging.info("Modèle LightGBM sauvegardé dans lightgbm_model.txt")

# Entraînement du modèle K-Means pour la segmentation
X_cluster = data[variables_clustering].dropna()
X_cluster_scaled = scaler.transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42)  
kmeans.fit(X_cluster_scaled)
joblib.dump(kmeans, 'kmeans_model.pkl')
logging.info("Modèle K-Means sauvegardé dans kmeans_model.pkl")
