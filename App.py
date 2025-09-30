# app.py
import streamlit as st
import eda, clustering, classification, deployment  # Ajout du module deployment

st.set_page_config(page_title="Analyse USAD Drépanocytose", layout="wide")

# ============================
# Barre latérale de navigation
# ============================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", [
    "Chapitre 1 : Cadre théorique",
    "Chapitre 2 : Analyse exploratoire",
    "Chapitre 3 : Classification non supervisée",
    "Chapitre 4 : Classification supervisée",
    "Déploiement du modèle"
])

# ============================
# Chapitre 1 : Cadre théorique
# ============================
if page == "Chapitre 1 : Cadre théorique":
    st.title("Cadre théorique et conceptuel")
    st.markdown("""
    - Présentation de l’USAD  
    - Généralités sur la drépanocytose  
    - Principes de l’intelligence artificielle appliquée à la santé
    """)

# ============================
# Chapitre 2 : Analyse exploratoire
# ============================
elif page == "Chapitre 2 : Analyse exploratoire":
    st.title("Analyse exploratoire des données")
    eda.show_eda()

# ============================
# Chapitre 3 : Classification non supervisée
# ============================
elif page == "Chapitre 3 : Classification non supervisée":
    clustering.show_clustering()

# ============================
# Chapitre 4 : Classification supervisée
# ============================
elif page == "Chapitre 4 : Classification supervisée":
    classification.show_classification()

# ============================
# Déploiement du modèle
# ============================
elif page == "Déploiement du modèle":
    
    deployment.show_deployment()  
