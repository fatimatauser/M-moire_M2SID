import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configuration du thème 
st.set_page_config(page_title="Prédiction des Urgences Drépanocytaires - USAD", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stSlider { color: #2196F3; }
    </style>
    """, unsafe_allow_html=True)

# Fonction pour charger les données 
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Format de fichier non supporté. Veuillez uploader un CSV ou Excel.")
                return None
            st.success(f"Fichier {uploaded_file.name} chargé avec succès ! ({len(df)} lignes)")
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            return None
    else:
        st.info("Aucun fichier uploadé. Utilisez l'uploader ci-dessous.")
        return None

# Uploader global dans la sidebar pour charger les données
st.sidebar.title("Télécharger Vos Données")
uploaded_file = st.sidebar.file_uploader("Uploader un fichier CSV ou Excel", type=['csv', 'xlsx', 'xls'])

df = load_data(uploaded_file)

# Chargement du modèle Random Forest 
try:
    model_rf = joblib.load('model_rf.joblib')  # Remplacez par votre fichier modèle
except:
    st.error("Modèle non trouvé. Entraînez et sauvegardez votre Random Forest avec joblib.dump(model, 'model_rf.joblib')")

# Préprocesseur 
encoder = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

# Sidebar pour navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélectionnez une section", ["Accueil", "Analyse Exploratoire", "Segmentation des Patients", "Prédiction des Risques", "À Propos"])

if page == "Accueil":
    st.title("Application Interactive pour l'Analyse et Prédiction des Urgences Drépanocytaires")
    st.markdown("""
    Bienvenue ! Cette application, développée dans le cadre d'un mémoire sur l'USAD, permet de :
    - Télécharger vos données (CSV/Excel) via la sidebar.
    - Visualiser les données cliniques.
    - Segmenter les patients via clustering (K-Means).
    - Prédire l'évolution des urgences (favorable ou avec complications) via Random Forest.
    Testée pour l'USAD – Contactez-moi pour des ajustements.
    """)
    st.image("logo_usad.png", width=200)  # Ajoutez un logo si disponible

elif page == "Analyse Exploratoire":
    st.title("Analyse Exploratoire des Données (EDA)")
    if df is not None:
        # Analyse univariée 
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Répartition par Sexe")
            fig_sex = px.pie(df, names='Sexe', title='Répartition par Sexe', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_sex)
        
        with col2:
            st.subheader("Distribution des Âges")
            fig_age = px.histogram(df, x='Âge du debut d etude en mois (en janvier 2023)', title='Distribution des Âges', color_discrete_sequence=['#2196F3'])
            st.plotly_chart(fig_age)
        
        # Analyse bivariée 
        st.subheader("Type de Drépanocytose vs Évolution")
        crosstab = pd.crosstab(df['Type de drépanocytose'], df['Evolution'])
        st.table(crosstab)
        fig_biv = px.bar(crosstab, title='Type de Drépanocytose vs Évolution', barmode='stack')
        st.plotly_chart(fig_biv)
    else:
        st.warning("Veuillez uploader un fichier de données pour afficher l'analyse.")

elif page == "Segmentation des Patients":
    st.title("Segmentation des Patients (Clustering Non Supervisé)")
    if df is not None:
        # Préparation des données pour clustering 
        features_cluster = ['Âge du debut d etude en mois (en janvier 2023)', 'Taux d\'Hb (g/dL)', '% d\'Hb F', '% d\'Hb S', 'Nbre de GB (/mm3)', 'Nbre de PLT (/mm3)']  # Ajoutez vos variables
        X_cluster = df[features_cluster].dropna()
        if len(X_cluster) > 0:
            X_scaled = scaler.fit_transform(X_cluster)
            
            # K-Means avec 3 clusters 
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            df_cluster = pd.DataFrame(X_scaled, columns=features_cluster)
            df_cluster['Cluster'] = clusters
            
            # Visualisation avec PCA
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(X_scaled)
            df_pca = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
            df_pca['Cluster'] = clusters
            
            st.subheader("Visualisation des Clusters (PCA)")
            fig_cluster = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster', title='Clusters de Patients', color_continuous_scale='Viridis')
            st.plotly_chart(fig_cluster)
            
            st.subheader("Profils des Clusters")
            st.table(df_cluster.groupby('Cluster').mean())
        else:
            st.warning("Pas de données valides pour le clustering.")
    else:
        st.warning("Veuillez uploader un fichier de données pour la segmentation.")

elif page == "Prédiction des Risques":
    st.title("Prédiction de l'Évolution des Urgences (Random Forest)")
    
    # Inputs utilisateur pour prédiction 
    st.subheader("Saisissez les Données du Patient")
    mois = st.selectbox("Mois de Consultation", ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'])  # Ajustez selon vos données
    hb = st.slider("Taux d'Hb (g/dL)", min_value=5.0, max_value=13.0, value=8.0)
    gb = st.slider("Nbre de GB (/mm3)", min_value=2000, max_value=20000, value=15000)
    plt_count = st.slider("Nbre de PLT (/mm3)", min_value=100000, max_value=900000, value=400000)
    paleur = st.checkbox("Présence de Pâleur")
    hospitalisation = st.checkbox("Prise en Charge Hospitalisation")
    # Ajoutez d'autres inputs basés sur vos variables retenues (ex. : Diagnostic Catégorisé, etc.)
    
    # Préparation des inputs pour le modèle
    input_data = pd.DataFrame({
        'Mois': [mois],
        'HB (g/dl)': [hb],
        'GB (/mm3)': [gb],
        'PLT (/mm3)': [plt_count],
        'Pâleur': [1 if paleur else 0],
        'Prise en charge Hospitalisation': [1 if hospitalisation else 0],
        # Ajoutez les autres variables avec des valeurs par défaut si nécessaire
    })
    
    # Prétraitement 
    input_encoded = pd.get_dummies(input_data)  # One-Hot pour catégorielles
    # Assurez-vous que les colonnes correspondent à celles de l'entraînement (ajoutez des colonnes manquantes à 0)
    # Exemple : input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)  # Assurez-vous que scaler est fitté sur les mêmes features
    
    if st.button("Prédire l'Évolution"):
        prediction_proba = model_rf.predict_proba(input_scaled)[0][1]  # Probabilité de complications (classe 1)
        prediction = "Complications" if prediction_proba > 0.56 else "Favorable"  # Seuil optimal de votre mémoire
        st.success(f"Prédiction : {prediction} (Probabilité de complications : {prediction_proba:.2f})")

    # Option pour prédire sur le dataset uploadé entier
    if df is not None and st.checkbox("Prédire sur l'ensemble des données uploadées"):
        # Prétraitez df comme pour l'entraînement (encodage, scaling)
        features_pred = ['Mois', 'HB (g/dl)', 'GB (/mm3)', 'PLT (/mm3)', 'Pâleur', 'Prise en charge Hospitalisation']  # Ajustez
        X_pred = df[features_pred].dropna()
        if len(X_pred) > 0:
            X_encoded = pd.get_dummies(X_pred)
            X_scaled = scaler.transform(X_encoded)
            predictions = model_rf.predict(X_scaled)
            df['Prediction'] = predictions
            st.subheader("Résultats des Prédictions")
            st.dataframe(df)
            st.download_button("Télécharger les Prédictions (CSV)", df.to_csv(index=False), file_name="predictions.csv")
        else:
            st.warning("Pas de données valides pour la prédiction.")

elif page == "À Propos":
    st.title("À Propos")
    st.markdown("""
    - **Développé par :** [Votre Nom] pour le mémoire sur l'USAD.
    - **Modèle :** Random Forest (Accuracy 98.4%, AUC 99.7%).
    - **Données :** Téléchargez votre CSV/Excel via la sidebar.
    - **Test pour USAD :** Exécutez localement ou déployez sur Streamlit Sharing. Contactez-moi pour accès.
    """)

# Footer
st.markdown("---")
st.markdown("Application v1.1 | Date : Septembre 2025 | Pour tests USAD avec upload de données")
