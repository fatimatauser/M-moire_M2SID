import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import joblib
import logging
import re
import warnings
import io

# Configuration des avertissements et logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration de la page pour un design large et thème personnalisé
st.set_page_config(
    page_title="Analyse de la Drépanocytose",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne et professionnel
st.markdown("""
    <style>
    /* Fond et couleurs principales */
    .stApp {
        background: linear-gradient(to bottom right, #e6f0fa, #ffffff);
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Titres stylés */
    h1, h2, h3 {
        color: #007BFF;
        font-weight: 600;
    }
    
    /* Boutons modernes */
    .stButton > button {
        background-color: #007BFF;
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 12px rgba(0, 91, 187, 0.3);
    }
    
    /* Sidebar élégante */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 3px solid #007BFF;
        padding: 20px;
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 16px;
        padding: 10px;
        border-radius: 8px;
        transition: background-color 0.3s;
    }
    [data-testid="stSidebar"] .stRadio > label:hover {
        background-color: #e6f0fa;
    }
    
    /* Containers avec ombre */
    .stMarkdown, .stDataFrame {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Animation d'entrée */
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    .slide-in {
        animation: slideIn 0.5s ease-in-out;
    }
    
    /* Progress bar et sliders */
    .stSlider > div > div > div {
        background-color: #007BFF;
    }
    
    /* File uploader */
    .stFileUploader > div > button {
        background-color: #28a745;
        color: white;
        border-radius: 8px;
    }
    .stFileUploader > div > button:hover {
        background-color: #218838;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal avec animation
st.markdown('<h1 class="slide-in">🩺 Analyse et Prédiction des Urgences Drépanocytaires</h1>', unsafe_allow_html=True)
st.markdown("""
    <p class="slide-in" style="color: #34495e;">
    Application interactive pour le mémoire de Master 2 en Santé : <b>Analyse et prédiction de l’évolution des urgences drépanocytaires chez les enfants à l'USAD</b>.  
    Chargez votre fichier Excel pour explorer les données, segmenter les patients et prédire les risques.
    </p>
""", unsafe_allow_html=True)

# Chargement des données via file uploader
st.markdown("### 📂 Charger la Base de Données")
uploaded_file = st.file_uploader("Sélectionnez un fichier Excel (.xlsx)", type=["xlsx"])
data = pd.DataFrame()

if uploaded_file is not None:
    try:
        # Lire toutes les feuilles du fichier Excel
        df = pd.read_excel(uploaded_file, sheet_name=None)
        data = pd.concat(df.values(), ignore_index=True)
        st.success(f"Fichier chargé avec succès : **{data.shape[0]} lignes**, **{data.shape[1]} colonnes**")
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        logging.error(f"Erreur chargement fichier : {e}")

# Chargement des modèles (assumez sauvegardés)
try:
    model = lgb.Booster(model_file='lightgbm_model.txt')
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
except:
    st.error("Modèles ou scaler non trouvés. Veuillez sauvegarder lightgbm_model.txt, scaler.pkl et kmeans_model.pkl.")
    logging.error("Modèles ou scaler non trouvés.")
    model, scaler, kmeans = None, None, None

# Variables utilisées pour la prédiction 
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

# Variables pour clustering
variables_clustering = [
    'Âge de début des signes (en mois)', 'Taux d\'Hb (g/dL)', '% d\'Hb F', 'Nbre de GB (/mm3)',
    'Nbre de PLT (/mm3)', 'Âge de découverte de la drépanocytose (en mois)'
]

# Vérification des colonnes nécessaires
required_columns = list(set(variables_selection + variables_clustering))
if not data.empty:
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.warning(f"Colonnes manquantes dans le fichier chargé : {', '.join(missing_columns)}")
        logging.warning(f"Colonnes manquantes : {missing_columns}")

# Fonctions utilitaires
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

# Sidebar avec navigation
st.sidebar.markdown('<h2 style="color: #007BFF;">🧭 Navigation</h2>', unsafe_allow_html=True)
st.sidebar.markdown("---")
pages = ["🏠 Accueil", "📊 Analyse Descriptive", "👥 Segmentation des Patients", "🔮 Prédiction des Urgences", "📈 Visualisations Interactives"]
page = st.sidebar.radio("", pages)

# Pages
if page == "🏠 Accueil":
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h2 class="slide-in">Bienvenue</h2>', unsafe_allow_html=True)
        st.markdown("""
            <div class="slide-in" style="color: #34495e;">
            Cette application permet :
            <ul>
                <li>Analyser les données des urgences drépanocytaires de 2023 à l'USAD.</li>
                <li>Segmenter les patients en clusters pour une prise en charge personnalisée.</li>
                <li>Prédire l'évolution des urgences (favorable ou complications).</li>
                <li>Visualiser les tendances avec des graphiques interactifs.</li>
            </ul>
            Chargez votre fichier Excel pour commencer l'analyse.
            </div>
        """, unsafe_allow_html=True)
        if not data.empty:
            st.info(f"Données chargées : **{data.shape[0]} lignes**, **{data.shape[1]} colonnes**")
    with col2:
        st.image("https://source.unsplash.com/random/300x200/?hospital", caption="Santé et Innovation", use_column_width=True)

elif page == "📊 Analyse Descriptive":
    if data.empty:
        st.warning("Veuillez charger un fichier Excel pour afficher l'analyse descriptive.")
    else:
        st.markdown('<h2 class="slide-in">Analyse Descriptive</h2>', unsafe_allow_html=True)
        tabs = st.tabs(["📈 Statistiques", "📊 Graphiques", "📋 Données Brutes"])
        
        with tabs[0]:
            st.subheader("Statistiques Générales")
            st.dataframe(data.describe().style.background_gradient(cmap='Blues'))
        
        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                if 'Sexe' in data.columns:
                    fig_sex = px.pie(data, names='Sexe', title='Répartition par Sexe', hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_sex, use_container_width=True)
            
            with col2:
                if 'DATE URGENCE 1' in data.columns:
                    data['Mois'] = pd.to_datetime(data['DATE URGENCE 1'], errors='coerce').dt.month_name()
                    fig_month = px.bar(data['Mois'].value_counts().sort_index(), title='Répartition Mensuelle des Urgences', 
                                     color_discrete_sequence=['#007BFF'])
                    st.plotly_chart(fig_month, use_container_width=True)
        
        with tabs[2]:
            st.subheader("Aperçu des Données")
            st.dataframe(data.head(10).style.set_table_styles([
                {'selector': 'tr:hover', 'props': [('background-color', '#e6f0fa')]}
            ]))

elif page == "👥 Segmentation des Patients":
    if data.empty:
        st.warning("Veuillez charger un fichier Excel pour effectuer la segmentation.")
    elif not all(col in data.columns for col in variables_clustering):
        st.error(f"Colonnes manquantes pour le clustering : {[col for col in variables_clustering if col not in data.columns]}")
    else:
        st.markdown('<h2 class="slide-in">Segmentation des Patients</h2>', unsafe_allow_html=True)
        X_cluster = data[variables_clustering].dropna()
        X_scaled = scaler.transform(X_cluster)
        clusters = kmeans.predict(X_scaled)
        data['Cluster'] = pd.Series(clusters, index=X_cluster.index)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Résumé des Clusters")
            st.dataframe(data.groupby('Cluster')[variables_clustering].mean().style.background_gradient(cmap='coolwarm'))
        
        with col2:
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            df_pca = pd.DataFrame({'PCA1': components[:,0], 'PCA2': components[:,1], 'Cluster': clusters})
            fig_pca = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', 
                               title='Visualisation des Clusters via PCA', 
                               color_continuous_scale='Viridis',
                               labels={'PCA1': f'PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                                       'PCA2': f'PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'})
            st.plotly_chart(fig_pca, use_container_width=True)

elif page == "🔮 Prédiction des Urgences":
    st.markdown('<h2 class="slide-in">Prédiction de l\'Évolution des Urgences</h2>', unsafe_allow_html=True)
    if model is None:
        st.error("Modèle LightGBM non chargé. Veuillez vérifier les fichiers requis.")
    else:
        with st.expander("Entrez les Données du Patient", expanded=True):
            st.markdown("### Informations Cliniques et Biologiques")
            col1, col2, col3 = st.columns(3)
            
            # Inputs pour les variables quantitatives
            with col1:
                age_debut = st.slider("Âge de début des signes (mois)", 0, 200, 25)
                age_decouverte = st.slider("Âge de découverte (mois)", 0, 200, 42)
                hb = st.number_input("Taux d'Hb (g/dL)", 0.0, 20.0, 8.0)
                hb_f = st.number_input("% d'Hb F", 0.0, 100.0, 20.0)
                hb_s = st.number_input("% d'Hb S", 0.0, 100.0, 76.0)
                gr = st.number_input("GR (/mm3)", 0.0, 10_000_000.0, 4_000_000.0)
            
            with col2:
                gb = st.number_input("Nbre de GB (/mm3)", 0.0, 100_000.0, 15_000.0)
                plt_count = st.number_input("Nbre de PLT (/mm3)", 0.0, 1_000_000.0, 400_000.0)
                vgm = st.number_input("VGM (fl/u3)", 0.0, 150.0, 80.0)
                tcmh = st.number_input("TCMH (g/dl)", 0.0, 50.0, 30.0)
                crp = st.number_input("CRP Si positive (Valeur)", 0.0, 500.0, 0.0)
                hosp_avant_2017 = st.number_input("Hospitalisations avant 2017", 0, 50, 0)
            
            with col3:
                hosp_2017_2023 = st.number_input("Hospitalisations 2017-2023", 0, 50, 0)
                trans_avant_2017 = st.number_input("Transfusions avant 2017", 0, 50, 0)
                trans_2017_2023 = st.number_input("Transfusions 2017-2023", 0, 50, 0)
                age_etude = st.slider("Âge à l'étude (mois, janvier 2023)", 0, 200, 60)
            
            # Inputs pour les variables qualitatives
            st.markdown("### Variables Catégorielles")
            col4, col5 = st.columns(2)
            with col4:
                paleur = st.selectbox("Pâleur", ["OUI", "NON"])
                souffle = st.selectbox("Souffle systolique fonctionnel", ["OUI", "NON"])
                vaccin_meningo = st.selectbox("Vaccin contre méningocoque", ["OUI", "NON"])
                vaccin_pneumo = st.selectbox("Vaccin contre pneumocoque", ["OUI", "NON"])
                prophylaxie = st.selectbox("Prophylaxie à la pénicilline", ["OUI", "NON"])
            
            with col5:
                splenomegalie = st.selectbox("Splénomégalie", ["OUI", "NON"])
                parents_salaries = st.selectbox("Parents Salariés", ["OUI", "NON"])
                hospitalisation = st.selectbox("Prise en charge Hospitalisation", ["OUI", "NON"])
                radio_thorax = st.selectbox("Radiographie du thorax", ["OUI", "NON"])
                douleur = st.selectbox("Douleur provoquée (Os/Abdomen)", ["OUI", "NON"])
            
            # Autres variables catégorielles
            niveau_urgence = st.selectbox("Niveau d'Urgence", ["Urgence1", "Urgence2", "Urgence3", "Urgence4", "Urgence5", "Urgence6"])
            diag_cat = st.selectbox("Diagnostic Catégorisé", ["CVO", "Infections", "Anémie", "STA", "AVC", "Autres"])
            mois = st.selectbox("Mois", ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"])
            scolarite = st.selectbox("Niveau d'instruction scolarité", ["Maternelle", "Elémentaire", "Secondaire", "Enseignement Supérieur", "NON"])
            hdj = st.selectbox("HDJ", ["OUI", "NON"])
            
            if st.button("🔍 Prédire l'Évolution", key="predict_button"):
                # Préparer les données d'entrée
                input_data = {
                    'Âge de début des signes (en mois)': age_debut,
                    'Âge de découverte de la drépanocytose (en mois)': age_decouverte,
                    'Taux d\'Hb (g/dL)': hb,
                    '% d\'Hb F': hb_f,
                    '% d\'Hb S': hb_s,
                    'GR (/mm3)': gr,
                    'Nbre de GB (/mm3)': gb,
                    'Nbre de PLT (/mm3)': plt_count,
                    'VGM (fl/u3)': vgm,
                    'TCMH (g/dl)': tcmh,
                    'CRP Si positive (Valeur)': crp,
                    'Nbre d\'hospitalisations avant 2017': hosp_avant_2017,
                    'Nbre d\'hospitalisations entre 2017 et 2023': hosp_2017_2023,
                    'Nbre de transfusion avant 2017': trans_avant_2017,
                    'Nbre de transfusion Entre 2017 et 2023': trans_2017_2023,
                    'Âge du debut d etude en mois (en janvier 2023)': age_etude,
                    'Pâleur': 1 if paleur == "OUI" else 0,
                    'Souffle systolique fonctionnel': 1 if souffle == "OUI" else 0,
                    'Vaccin contre méningocoque': 1 if vaccin_meningo == "OUI" else 0,
                    'Vaccin contre pneumocoque': 1 if vaccin_pneumo == "OUI" else 0,
                    'Prophylaxie à la pénicilline': 1 if prophylaxie == "OUI" else 0,
                    'Splénomégalie': 1 if splenomegalie == "OUI" else 0,
                    'Parents Salariés': 1 if parents_salaries == "OUI" else 0,
                    'Prise en charge Hospitalisation': 1 if hospitalisation == "OUI" else 0,
                    'Radiographie du thorax Oui ou Non': 1 if radio_thorax == "OUI" else 0,
                    'Douleur provoquée (Os.Abdomen)': 1 if douleur == "OUI" else 0,
                    'NiveauUrgence': {'Urgence1': 1, 'Urgence2': 2, 'Urgence3': 3, 'Urgence4': 4, 'Urgence5': 5, 'Urgence6': 6}[niveau_urgence],
                    "Niveau d'instruction scolarité": {'Maternelle': 1, 'Elémentaire': 2, 'Secondaire': 3, 'Enseignement Supérieur': 4, 'NON': 0}[scolarite],
                    'HDJ': 1 if hdj == "OUI" else 0
                }
                
                # Ajouter les colonnes one-hot pour Diagnostic Catégorisé et Mois
                diag_cols = ['Diagnostic Catégorisé_Infections', 'Diagnostic Catégorisé_Anémie', 'Diagnostic Catégorisé_STA', 'Diagnostic Catégorisé_AVC', 'Diagnostic Catégorisé_Autres']
                mois_cols = [f'Mois_{m}' for m in ['Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']]
                for col in diag_cols:
                    input_data[col] = 1 if col == f'Diagnostic Catégorisé_{diag_cat}' else 0
                for col in mois_cols:
                    input_data[col] = 1 if col == f'Mois_{mois}' else 0
                
                # Créer DataFrame pour l'entrée
                input_df = pd.DataFrame([input_data])
                
                # Standardiser les variables quantitatives
                quantitative_vars = [
                    'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
                    'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
                    'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
                    'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
                    "Nbre d'hospitalisations entre 2017 et 2023",
                    'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
                    'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", '% d\'Hb F'
                ]
                input_df[quantitative_vars] = scaler.transform(input_df[quantitative_vars])
                
                # Vérifier l'ordre des colonnes
                input_df = input_df[model.feature_name()]
                
                # Prédiction
                proba = model.predict(input_df)[0]
                prediction = "Complications" if proba >= 0.998 else "Favorable"
                
                st.markdown("### Résultat de la Prédiction")
                st.success(f"**Probabilité de complications :** {proba:.2f}")
                st.info(f"**Évolution prédite :** {prediction}")
                
                # Jauge pour visualisation
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba*100,
                    title={'text': "Risque de Complications (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "red" if proba > 0.5 else "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "#e6f0fa"},
                            {'range': [50, 100], 'color': "#ffcccc"}
                        ]
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

elif page == "📈 Visualisations Interactives":
    if data.empty:
        st.warning("Veuillez charger un fichier Excel pour afficher les visualisations.")
    elif model is None:
        st.error("Modèle LightGBM non chargé. Veuillez vérifier les fichiers requis.")
    else:
        st.markdown('<h2 class="slide-in">Visualisations Interactives</h2>', unsafe_allow_html=True)
        feature_importances = pd.DataFrame({
            'Feature': model.feature_name(),
            'Importance': model.feature_importance(importance_type='gain')
        }).sort_values(by='Importance', ascending=False)
        
        fig_imp = px.bar(feature_importances, x='Importance', y='Feature', orientation='h', 
                        title='Importance des Variables (LightGBM)', 
                        color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Courbe ROC (placeholder)
        st.subheader("Courbe ROC (Exemple)")
        fpr, tpr = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC (AUC=0.996)', line=dict(color='#007BFF')))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aléatoire', line=dict(dash='dash', color='grey')))
        fig_roc.update_layout(title='Courbe ROC', xaxis_title='Taux de Faux Positifs', yaxis_title='Taux de Vrais Positifs')
        st.plotly_chart(fig_roc, use_container_width=True)

# Pied de page
st.markdown("---")
st.markdown("""
    <p style="text-align: center; color: #666; font-size: 14px;">
    Développé par FATIMATA KANE & ISSEU GUEYE pour le Mémoire de Master 2 Santé | © 2025 | Contact: email@gmail.com
    </p>
""", unsafe_allow_html=True)
