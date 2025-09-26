import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
import joblib

# ================================
# Chargement des données
# ================================
# Mettez à jour ce chemin avec l'emplacement exact de votre fichier
df = pd.read_excel(r"fichier_nettoye.xlsx")  

# ================================
# Variables sélectionnées
# ================================
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
    'HDJ', 'Nbre de transfusion Entre 2017 et 2023', 'Evolution'
]
# Gestion des colonnes manquantes
for col in variables_selection:
    if col not in df.columns:
        df[col] = 0 if col in ['Pâleur', 'Souffle systolique fonctionnel', 'Vaccin contre méningocoque',
                               'Splénomégalie', 'Prophylaxie à la pénicilline', 'Parents Salariés',
                               'Prise en charge Hospitalisation', 'Radiographie du thorax Oui ou Non',
                               'Douleur provoquée (Os.Abdomen)', 'Vaccin contre pneumocoque'] \
                  else np.nan if col in ['Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
                                         'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
                                         'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
                                         'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
                                         "Nbre d'hospitalisations entre 2017 et 2023",
                                         'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
                                         'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"] \
                  else 'NON'
df_selected = df[variables_selection].copy()

# ================================
# Encodage
# ================================
binary_mappings = {
    'Pâleur': {'OUI':1, 'NON':0},
    'Souffle systolique fonctionnel': {'OUI':1, 'NON':0},
    'Vaccin contre méningocoque': {'OUI':1, 'NON':0},
    'Splénomégalie': {'OUI':1, 'NON':0},
    'Prophylaxie à la pénicilline': {'OUI':1, 'NON':0},
    'Parents Salariés': {'OUI':1, 'NON':0},
    'Prise en charge Hospitalisation': {'OUI':1, 'NON':0},
    'Radiographie du thorax Oui ou Non': {'OUI':1, 'NON':0},
    'Douleur provoquée (Os.Abdomen)': {'OUI':1, 'NON':0},
    'Vaccin contre pneumocoque': {'OUI':1, 'NON':0},
}
df_selected.replace(binary_mappings, inplace=True)

ordinal_mappings = {
    'NiveauUrgence': {'Urgence1':1, 'Urgence2':2, 'Urgence3':3, 'Urgence4':4, 'Urgence5':5, 'Urgence6':6},
    "Niveau d'instruction scolarité": {'Maternelle ':1, 'Elémentaire ':2, 'Secondaire':3, 'Enseignement Supérieur ':4, 'NON':0}
}
df_selected.replace(ordinal_mappings, inplace=True)

# One-Hot Encoding pour les variables catégorielles
df_selected = pd.get_dummies(df_selected, columns=['Diagnostic Catégorisé', 'Mois'], drop_first=True)

# ================================
# Standardisation
# ================================
quantitative_vars = [
    'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
    'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
    'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
    'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
    "Nbre d'hospitalisations entre 2017 et 2023",
    'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
    'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"
]
scaler = StandardScaler()
df_selected[quantitative_vars] = scaler.fit_transform(df_selected[quantitative_vars])

# ================================
# Variable cible
# ================================
df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})
X = df_selected.drop(['Evolution', 'Evolution_Cible'], axis=1)
y = df_selected['Evolution_Cible']

# ================================
# SMOTETomek
# ================================
print("Avant SMOTETomek :")
print(y.value_counts())

smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X, y)

print("\nAprès SMOTETomek :")
print(pd.Series(y_res).value_counts())

# ================================
# Division train/val/test
# ================================
X_train, X_temp, y_train, y_temp = train_test_split(X_res, y_res, test_size=0.4, stratify=y_res, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# ================================
# Entraînement du modèle Random Forest
# ================================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Évaluation rapide
y_test_proba = rf_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
print("\nRandom Forest - Rapport de classification :")
print(classification_report(y_test, y_test_pred))
print(f"AUC-ROC : {roc_auc_score(y_test, y_test_proba)}")
print(f"Seuil optimal : {optimal_threshold}")

# ================================
# Sauvegarde du modèle, scaler et features
# ================================
joblib.dump(rf_model, r"C:\Users\FATIMATA\usad-streamlit\model_rf.joblib")
joblib.dump(scaler, r"C:\Users\FATIMATA\usad-streamlit\scaler.joblib")
features = X_train.columns.tolist()
joblib.dump(features, r"C:\Users\FATIMATA\usad-streamlit\features.joblib")
print("Modèle, scaler et features sauvegardés avec succès dans :")
print("- C:\\Users\\FATIMATA\\Desktop\\M2SID\\Mémoire\\application\\model_rf.joblib")
print("- C:\\Users\\FATIMATA\\Desktop\\M2SID\\Mémoire\\application\\scaler.joblib")
print("- C:\\Users\\FATIMATA\\Desktop\\M2SID\\Mémoire\\application\\features.joblib")
