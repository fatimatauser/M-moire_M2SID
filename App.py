import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configuration du thème 
st.set_page_config(page_title="USAD - Prédiction des Urgences Drépanocytaires", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stSelectbox, .stNumberInput, .stCheckbox {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSidebar .stRadio > div {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 10px;
    }
    h1, h2, h3 { color: #003087; font-family: 'Arial', sans-serif; }
    .stMarkdown { font-family: 'Arial', sans-serif; }
    .sidebar .sidebar-content { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# Liste des variables quantitatives
quantitative_vars = [
    'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
    'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
    'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
    'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
    "Nbre d'hospitalisations entre 2017 et 2023",
    'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
    'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"
]

# Mappages pour l'encodage
binary_mappings = {
    'Pâleur': {'OUI': 1, 'NON': 0},
    'Souffle systolique fonctionnel': {'OUI': 1, 'NON': 0},
    'Vaccin contre méningocoque': {'OUI': 1, 'NON': 0},
    'Splénomégalie': {'OUI': 1, 'NON': 0},
    'Prophylaxie à la pénicilline': {'OUI': 1, 'NON': 0},
    'Parents Salariés': {'OUI': 1, 'NON': 0},
    'Prise en charge Hospitalisation': {'OUI': 1, 'NON': 0},
    'Radiographie du thorax Oui ou Non': {'OUI': 1, 'NON': 0},
    'Douleur provoquée (Os.Abdomen)': {'OUI': 1, 'NON': 0},
    'Vaccin contre pneumocoque': {'OUI': 1, 'NON': 0},
}

ordinal_mappings = {
    'NiveauUrgence': {'Urgence1': 1, 'Urgence2': 2, 'Urgence3': 3, 'Urgence4': 4, 'Urgence5': 5, 'Urgence6': 6},
    "Niveau d'instruction scolarité": {'Maternelle ': 1, 'Elémentaire ': 2, 'Secondaire': 3, 'Enseignement Supérieur ': 4, 'NON': 0}
}

# Fonction pour charger et prétraiter les données
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Format de fichier non supporté. Veuillez uploader un CSV ou Excel.")
                return None
            st.success(f"✅ Fichier {uploaded_file.name} chargé avec succès ! ({len(df)} lignes)")
            
            # Sélection des variables pertinentes
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
            missing_cols = [col for col in variables_selection if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠️ Colonnes manquantes dans le fichier : {missing_cols}. Elles seront ajoutées avec des valeurs par défaut.")
                for col in missing_cols:
                    df[col] = 0 if col in binary_mappings else np.nan if col in quantitative_vars else 'NON'
            
            df = df[variables_selection].copy()
            
            # Nettoyage des variables quantitatives (gérer les virgules et espaces)
            for var in quantitative_vars:
                if var in df.columns:
                    df[var] = df[var].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
            
            # Encodage
            df.replace(binary_mappings, inplace=True)
            df.replace(ordinal_mappings, inplace=True)
            df = pd.get_dummies(df, columns=['Diagnostic Catégorisé', 'Mois'], drop_first=True)
            
            # Standardisation
            scaler = joblib.load('scaler.joblib')
            df[quantitative_vars] = scaler.transform(df[quantitative_vars])
            
            return df
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement/prétraitement du fichier : {e}")
            return None
    else:
        st.info("ℹ️ Aucun fichier uploadé. Utilisez l'uploader dans la barre latérale.")
        return None

# Uploader global dans la barre latérale
st.sidebar.header("📤 Télécharger Vos Données")
uploaded_file = st.sidebar.file_uploader("Choisir un fichier CSV ou Excel", type=['csv', 'xlsx', 'xls'], help="Formats supportés : CSV, XLSX, XLS")

# Charger les données
df = load_and_preprocess_data(uploaded_file)

# Charger le modèle et les features
try:
    model_rf = joblib.load('model_rf.joblib')
    features = joblib.load('features.joblib')
    scaler = joblib.load('scaler.joblib')
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des fichiers du modèle : {e}. Vérifiez que 'model_rf.joblib', 'scaler.joblib' et 'features.joblib' sont dans le répertoire.")
    st.stop()

# Barre latérale pour navigation
st.sidebar.header("🧭 Navigation")
page = st.sidebar.radio(
    "Choisir une section",
    ["🏠 Accueil", "📊 Analyse Exploratoire", "🗂 Segmentation des Patients", "🔍 Prédiction des Risques", "ℹ️ À Propos"],
    label_visibility="collapsed"
)

# En-tête global
with st.container():
    st.markdown("<h1 style='text-align: center; color: #003087;'>USAD - Prédiction des Urgences Drépanocytaires</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>Application pour l'analyse et la prédiction des crises drépanocytaires</h3>", unsafe_allow_html=True)

if page == "🏠 Accueil":
    with st.container():
        st.header("Bienvenue à l'Application USAD 🌟")
        st.markdown("""
        Cette application, développée dans le cadre d'un mémoire sur l'**Unité de Soins Ambulatoires des enfants et adolescents Drépanocytaires (USAD)**, permet de :
        - 📤 **Télécharger** vos données cliniques (CSV ou Excel) via la barre latérale.
        - 📊 **Visualiser** les tendances des données avec des graphiques interactifs.
        - 🗂 **Segmenter** les patients en groupes à l'aide du clustering (K-Means).
        - 🔍 **Prédire** l'évolution des urgences (favorable ou avec complications) grâce à un modèle Random Forest.
        Testée pour l'USAD Sénégal – Contactez-nous pour toute question ou ajustement.
        """)
        st.info("ℹ️ Commencez par uploader un fichier dans la barre latérale pour explorer les données !")

elif page == "📊 Analyse Exploratoire":
    st.header("📊 Analyse Exploratoire des Données (EDA)")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Répartition par Pâleur")
            fig_paleur = px.pie(df, names='Pâleur', title='Répartition par Pâleur', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_paleur, use_container_width=True)
        
        with col2:
            st.subheader("Distribution des Âges")
            fig_age = px.histogram(df, x='Âge du debut d etude en mois (en janvier 2023)', title='Distribution des Âges', color_discrete_sequence=['#007bff'])
            st.plotly_chart(fig_age, use_container_width=True)
        
        st.subheader("Niveau d'Urgence vs Évolution")
        crosstab = pd.crosstab(df['NiveauUrgence'], df['Evolution'])
        st.table(crosstab)
        fig_biv = px.bar(crosstab, title='Niveau d\'Urgence vs Évolution', barmode='stack', color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_biv, use_container_width=True)
    else:
        st.warning("⚠️ Veuillez uploader un fichier de données pour afficher l'analyse.")

elif page == "🗂 Segmentation des Patients":
    st.header("🗂 Segmentation des Patients (Clustering Non Supervisé)")
    if df is not None:
        features_cluster = [
            'Âge du debut d etude en mois (en janvier 2023)', "Taux d'Hb (g/dL)", '% d\'Hb F', '% d\'Hb S',
            'Nbre de GB (/mm3)', 'Nbre de PLT (/mm3)'
        ]
        X_cluster = df[features_cluster].dropna()
        if len(X_cluster) > 0:
            X_scaled = scaler.transform(X_cluster)
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            df_cluster = pd.DataFrame(X_scaled, columns=features_cluster)
            df_cluster['Cluster'] = clusters
            
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(X_scaled)
            df_pca = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
            df_pca['Cluster'] = clusters
            
            st.subheader("Visualisation des Clusters (PCA)")
            fig_cluster = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster', title='Clusters de Patients', color_continuous_scale='Viridis')
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            st.subheader("Profils des Clusters")
            st.table(df_cluster.groupby('Cluster').mean())
        else:
            st.warning("⚠️ Pas de données valides pour le clustering.")
    else:
        st.warning("⚠️ Veuillez uploader un fichier de données pour la segmentation.")

elif page == "🔍 Prédiction des Risques":
    st.header("🔍 Prédiction de l'Évolution des Urgences (Random Forest)")
    
    with st.form("prediction_form"):
        st.subheader("Saisissez les Données du Patient")
        col1, col2 = st.columns(2)
        input_data = {}
        for feature in [
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
        ]:
            if feature in quantitative_vars:
                with col1:
                    input_data[feature] = st.number_input(feature, value=0.0, step=0.1, format="%.1f")
            elif feature in binary_mappings:
                with col2:
                    input_data[feature] = st.selectbox(feature, options=['OUI', 'NON'])
            elif feature == 'NiveauUrgence':
                with col1:
                    input_data[feature] = st.selectbox(feature, options=['Urgence1', 'Urgence2', 'Urgence3', 'Urgence4', 'Urgence5', 'Urgence6'])
            elif feature == "Niveau d'instruction scolarité":
                with col2:
                    input_data[feature] = st.selectbox(feature, options=['Maternelle ', 'Elémentaire ', 'Secondaire', 'Enseignement Supérieur ', 'NON'])
            elif feature == 'Diagnostic Catégorisé':
                with col1:
                    input_data[feature] = st.selectbox(feature, options=df['Diagnostic Catégorisé'].unique() if df is not None else ['Unknown'])
            elif feature == 'Mois':
                with col2:
                    input_data[feature] = st.selectbox(feature, options=['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'])
        
        submitted = st.form_submit_button("🔍 Prédire l'Évolution")
        if submitted:
            input_df = pd.DataFrame([input_data])
            input_df.replace(binary_mappings, inplace=True)
            input_df.replace(ordinal_mappings, inplace=True)
            input_df = pd.get_dummies(input_df, columns=['Diagnostic Catégorisé', 'Mois'], drop_first=True)
            
            for col in features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[features]
            
            input_df[quantitative_vars] = scaler.transform(input_df[quantitative_vars])
            
            pred_proba = model_rf.predict_proba(input_df)[:, 1]
            pred_class = (pred_proba >= 0.56).astype(int)
            prediction = "Complications" if pred_class[0] == 1 else "Favorable"
            st.success(f"✅ Prédiction : {prediction} (Probabilité de complications : {pred_proba[0]:.2f})")

    if df is not None:
        with st.expander("Prédire sur l'ensemble des données uploadées"):
            if st.button("📈 Lancer la prédiction globale"):
                X_pred = df.drop(['Evolution'], axis=1, errors='ignore')
                for col in features:
                    if col not in X_pred.columns:
                        X_pred[col] = 0
                X_pred = X_pred[features]
                predictions = model_rf.predict(X_pred)
                predictions_proba = model_rf.predict_proba(X_pred)[:, 1]
                df['Prediction'] = predictions
                df['Probabilité Complications'] = predictions_proba
                st.subheader("Résultats des Prédictions")
                st.dataframe(df)
                st.download_button("📥 Télécharger les Prédictions (CSV)", df.to_csv(index=False), file_name="predictions.csv")

elif page == "ℹ️ À Propos":
    st.header("ℹ️ À Propos")
    with st.container():
        st.markdown("""
        **Développé par :** FATIMATA KANE & ISSEU GUEYE pour le mémoire de M2SID.  
        **Objectif :** Fournir un outil interactif pour l'analyse et la prédiction des urgences drépanocytaires.  
        **Modèle :** Random Forest.  
        **Données :** Téléchargez votre fichier CSV/Excel via la barre latérale.  
        **Test pour USAD :** Déployé via GitHub/Streamlit Cloud.   
        **Version :** 1.2 | **Date :** Septembre 2025
        """)

# Pied de page
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Application USAD v1.2 | Septembre 2025 | Développée pour l'USAD Sénégal</p>", unsafe_allow_html=True)
