# ================================
# classification.py pour Streamlit - version finale mémoire-ready
# ================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb

def show_classification():
    st.set_page_config(page_title="Classification Supervisée", layout="wide")
    st.markdown("<h1 style='text-align:center;color:darkblue;'>Classification Supervisée - Analyse des Modèles</h1>", unsafe_allow_html=True)

    # -------------------------------
    # 1️⃣ Chargement des données
    # -------------------------------
    df = pd.read_excel("fichier_nettoye.xlsx")
    st.markdown("### Aperçu des données")
    st.dataframe(df.head(), use_container_width=True)

    # -------------------------------
    # 2️⃣ Préparation des données
    # -------------------------------
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
    df_selected = df[variables_selection].copy()

    # Encodage binaire
    binary_mappings = {col: {'OUI':1,'NON':0} for col in [
        'Pâleur','Souffle systolique fonctionnel','Vaccin contre méningocoque',
        'Splénomégalie','Prophylaxie à la pénicilline','Parents Salariés',
        'Prise en charge Hospitalisation','Radiographie du thorax Oui ou Non',
        'Douleur provoquée (Os.Abdomen)','Vaccin contre pneumocoque']}
    df_selected.replace(binary_mappings, inplace=True)

    # Encodage ordinal
    ordinal_mappings = {
        'NiveauUrgence': {'Urgence1':1,'Urgence2':2,'Urgence3':3,'Urgence4':4,'Urgence5':5,'Urgence6':6},
        "Niveau d'instruction scolarité": {'Maternelle ':1,'Elémentaire ':2,'Secondaire':3,'Enseignement Supérieur ':4,'NON':0}
    }
    df_selected.replace(ordinal_mappings, inplace=True)

    # Variables catégorielles en dummies
    df_selected = pd.get_dummies(df_selected, columns=['Diagnostic Catégorisé','Mois'], drop_first=True)

    # Standardisation
    quantitative_vars = [
        'Âge de début des signes (en mois)','GR (/mm3)','GB (/mm3)',
        'Âge du debut d etude en mois (en janvier 2023)','VGM (fl/u3)','HB (g/dl)',
        'Nbre de GB (/mm3)','PLT (/mm3)','Nbre de PLT (/mm3)','TCMH (g/dl)',
        "Nbre d'hospitalisations avant 2017","Nbre d'hospitalisations entre 2017 et 2023",
        'Nbre de transfusion avant 2017','Nbre de transfusion Entre 2017 et 2023',
        'CRP Si positive (Valeur)',"Taux d'Hb (g/dL)","% d'Hb S","% d'Hb F"
    ]
    df_selected[quantitative_vars] = StandardScaler().fit_transform(df_selected[quantitative_vars])

    # Variable cible
    df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0,'Complications':1})
    X = df_selected.drop(['Evolution','Evolution_Cible'], axis=1)
    y = df_selected['Evolution_Cible']

    # SMOTETomek
    X_res, y_res = SMOTETomek(random_state=42).fit_resample(X, y)

    # Division train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X_res,y_res,test_size=0.4,stratify=y_res,random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.5,stratify=y_temp,random_state=42)

    # -------------------------------
    # 3️⃣ Modèles
    # -------------------------------
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "LightGBM": lgb.LGBMClassifier(objective='binary',learning_rate=0.05,num_leaves=31,n_estimators=500,random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:,1]
        fpr,tpr,thresholds = roc_curve(y_val,y_val_proba)
        optimal_threshold = thresholds[np.argmax(tpr-fpr)]
        y_test_proba = model.predict_proba(X_test)[:,1]
        y_test_pred = (y_test_proba>=optimal_threshold).astype(int)
        cm = confusion_matrix(y_test,y_test_pred)
        report = classification_report(y_test,y_test_pred,output_dict=True)
        auc = roc_auc_score(y_test,y_test_proba)
        results[name] = {"CM":cm,"Report":report,"AUC":auc,"Threshold":optimal_threshold,"Model":model}

    # -------------------------------
    # Résumé des métriques
    # -------------------------------
    summary = []
    for name,res in results.items():
        r = res["Report"]
        summary.append({
            "Modèle":name,
            "Accuracy":round(r['accuracy'],3),
            "Precision":round(r['macro avg']['precision'],3),
            "Recall":round(r['macro avg']['recall'],3),
            "F1":round(r['macro avg']['f1-score'],3),
            "AUC":round(res['AUC'],3)
        })
    summary_df = pd.DataFrame(summary)
    summary_df['Mean'] = summary_df[['Accuracy','Precision','Recall','F1','AUC']].mean(axis=1)
    best_name = summary_df.loc[summary_df['Mean'].idxmax(),'Modèle']
    best_model = results[best_name]["Model"]

    # -------------------------------
    # Onglets Streamlit
    # -------------------------------
    tab1, tab2, tab3 = st.tabs(["Comparaison des modèles","Méthodologie","Analyse du meilleur modèle"])

    # -------------------------------
    # Onglet 1 - Comparaison des modèles
    # -------------------------------
    with tab1:
        st.subheader("Comparaison des modèles")

        # 1️⃣ Matrice de confusion du meilleur modèle
        st.markdown(f"### Matrice de confusion du meilleur modèle : <span style='color:darkred;'>{best_name}</span>", unsafe_allow_html=True)
        st.write(results[best_name]["CM"])

        # 2️⃣ Barplot comparatif AUC + Precision
        fig_bar = px.bar(summary_df, x="Modèle", y=["AUC","Precision"],
                         barmode='group', title="Comparaison des modèles : AUC vs Precision",
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig_bar.update_layout(width=750, height=450)
        st.plotly_chart(fig_bar)

        # 3️⃣ Heatmap des métriques
        metrics_heatmap = summary_df.set_index('Modèle')[['Accuracy','Precision','Recall','F1','AUC']]
        heatmap_fig = ff.create_annotated_heatmap(
            z=np.round(metrics_heatmap.values,3),
            x=metrics_heatmap.columns.tolist(),
            y=metrics_heatmap.index.tolist(),
            colorscale='Viridis', showscale=True, reversescale=True
        )
        heatmap_fig.update_layout(title='Heatmap des métriques des modèles', width=750, height=500)
        st.plotly_chart(heatmap_fig)

    # -------------------------------
    # Onglet 2 - Méthodologie
    # -------------------------------
    with tab2:
        st.markdown("### Étapes méthodologiques")
        st.markdown("""
        1. Chargement et exploration des données brutes.
        2. Sélection des variables pertinentes.
        3. Encodage des variables binaires et ordinales.
        4. Création de variables factices pour les catégories.
        5. Standardisation des variables quantitatives.
        6. Définition de la variable cible et encodage.
        7. Gestion du déséquilibre avec SMOTETomek.
        8. Division en ensembles train/validation/test.
        9. Définition et entraînement de plusieurs modèles supervisés.
        10. Évaluation des modèles sur plusieurs métriques (Accuracy, Precision, Recall, F1, AUC).
        11. Comparaison visuelle des modèles.
        12. Sélection du meilleur modèle basé sur l’ensemble des métriques.
        """)

    # -------------------------------
    # Onglet 3 - Analyse du meilleur modèle
    # -------------------------------
    with tab3:
        st.markdown(f"### Analyse détaillée du meilleur modèle : <span style='color:darkgreen;'>{best_name}</span>", unsafe_allow_html=True)
        st.markdown(f"""
        - **Accuracy** : {summary_df.loc[summary_df['Modèle']==best_name,'Accuracy'].values[0]}
        - **Precision** : {summary_df.loc[summary_df['Modèle']==best_name,'Precision'].values[0]}
        - **Recall** : {summary_df.loc[summary_df['Modèle']==best_name,'Recall'].values[0]}
        - **F1-score** : {summary_df.loc[summary_df['Modèle']==best_name,'F1'].values[0]}
        - **AUC** : {summary_df.loc[summary_df['Modèle']==best_name,'AUC'].values[0]}
        """)

        # Courbe ROC
        y_test_proba = best_model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        auc_score = roc_auc_score(y_test, y_test_proba)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve', line=dict(color='darkblue', width=3)))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(color='red', width=2, dash='dash')))
        roc_fig.update_layout(title=f'Courbe ROC - {best_name} (AUC = {auc_score:.3f})', xaxis_title='FPR', yaxis_title='TPR', width=750, height=500)
        st.plotly_chart(roc_fig)

        # Variables importantes
        if hasattr(best_model, "feature_importances_"):
            importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(15)
            fig_imp = go.Figure(go.Bar(
                x=importances.values,
                y=importances.index,
                orientation='h',
                marker=dict(color=importances.values, colorscale='Viridis'),
            ))
            fig_imp.update_layout(title=f'Top 15 des variables importantes - {best_name}', xaxis_title='Importance', yaxis_title='Variables', width=750, height=600)
            st.plotly_chart(fig_imp)
        else:
            st.info("Pas de variable importance disponible pour ce modèle.")
