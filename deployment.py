# ================================
# deployment.py - Déploiement Random Forest
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

def show_deployment():
    st.set_page_config(page_title="Déploiement Random Forest", layout="wide")
    st.markdown("<h1 style='text-align:center;color:darkgreen;'>Déploiement - Random Forest</h1>", unsafe_allow_html=True)

    # -------------------------------
    # Charger le modèle et le scaler
    # -------------------------------
    model = joblib.load("random_forest_model.pkl")  # Modèle Random Forest sauvegardé
    scaler = joblib.load("scaler.pkl")             # Scaler utilisé pour les variables quantitatives

    # Variables quantitatives
    quantitative_vars = [
        'Âge de début des signes (en mois)','GR (/mm3)','GB (/mm3)',
        'Âge du debut d etude en mois (en janvier 2023)','VGM (fl/u3)','HB (g/dl)',
        'Nbre de GB (/mm3)','PLT (/mm3)','Nbre de PLT (/mm3)','TCMH (g/dl)',
        "Nbre d'hospitalisations avant 2017","Nbre d'hospitalisations entre 2017 et 2023",
        'Nbre de transfusion avant 2017','Nbre de transfusion Entre 2017 et 2023',
        'CRP Si positive (Valeur)',"Taux d'Hb (g/dL)","% d'Hb S","% d'Hb F"
    ]

    # Variables binaires
    binary_vars = [
        'Pâleur','Souffle systolique fonctionnel','Vaccin contre méningocoque',
        'Splénomégalie','Prophylaxie à la pénicilline','Parents Salariés',
        'Prise en charge Hospitalisation','Radiographie du thorax Oui ou Non',
        'Douleur provoquée (Os.Abdomen)','Vaccin contre pneumocoque'
    ]

    st.markdown("### Utilisez le formulaire ci-dessous pour prédire l’évolution d’un patient selon le modèle Random Forest.")

    # Extraire les catégories exactes du modèle entraîné
    model_features = model.feature_names_in_
    diagnostic_cols = [c for c in model_features if "Diagnostic Catégorisé_" in c]
    mois_cols = [c for c in model_features if "Mois_" in c]

    # Extraire les catégories originales pour le formulaire
    diagnostic_categories = [c.replace("Diagnostic Catégorisé_", "") for c in diagnostic_cols]
    mois_categories = [c.replace("Mois_", "") for c in mois_cols]

    with st.form("patient_form"):
        st.subheader("Variables quantitatives")
        quantitative_inputs = {}
        for var in quantitative_vars:
            quantitative_inputs[var] = st.number_input(var, value=0.0)

        st.subheader("Variables binaires (OUI=1, NON=0)")
        binary_inputs = {}
        for var in binary_vars:
            binary_inputs[var] = st.selectbox(var, options=[0,1])

        st.subheader("Variables ordinales")
        niveau_urgence = st.slider("Niveau d'urgence (1=Urgence1 ... 6=Urgence6)", 1, 6, 1)
        niveau_instruction = st.selectbox(
            "Niveau d'instruction scolarité (0=NON,1=Maternelle,2=Elémentaire,3=Secondaire,4=Supérieur)",
            options=[0,1,2,3,4]
        )

        st.subheader("Variables catégorielles")
        diagnostic = st.selectbox("Diagnostic Catégorisé", options=diagnostic_categories)
        mois = st.selectbox("Mois", options=mois_categories)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        # Préparer les données sous forme de DataFrame
        input_dict = {**quantitative_inputs, **binary_inputs,
                      'NiveauUrgence': niveau_urgence,
                      "Niveau d'instruction scolarité": niveau_instruction,
                      "Diagnostic Catégorisé": diagnostic,
                      "Mois": mois}

        input_df = pd.DataFrame([input_dict])

        # Encodage dummies identique à l'entraînement
        input_df = pd.get_dummies(input_df, columns=["Diagnostic Catégorisé","Mois"], drop_first=True)

        # Ajouter les colonnes manquantes avec 0
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0

        # Réordonner les colonnes pour correspondre au modèle
        input_df = input_df[model_features]

        # Standardisation des variables quantitatives
        input_df[quantitative_vars] = scaler.transform(input_df[quantitative_vars])

        # Prédiction
        pred_proba = model.predict_proba(input_df)[:,1][0]
        pred_class = model.predict(input_df)[0]

        # Résultat
        st.subheader("Résultat de la prédiction")
        if pred_class == 0:
            st.success(f"Évolution prévue : **Favorable** ✅ (Probabilité de complication : {pred_proba:.2f})")
        else:
            st.error(f"Évolution prévue : **Complications** ⚠️ (Probabilité : {pred_proba:.2f})")


