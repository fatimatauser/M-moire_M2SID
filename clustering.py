# clustering.py
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def show_clustering():
    st.title("Segmentation de Patients - KMeans Clustering")

    # ================================
    # 1. Chargement automatique de la base
    # ================================
    st.info("Chargement automatique de la base de données : segmentation.xlsx")
    df = pd.read_excel("segmentation.xlsx")
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # ================================
    # 2. Variables sélectionnées
    # ================================
    variables_selected = [
        "Âge du debut d etude en mois (en janvier 2023)",
        "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
        "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)",
        "Âge de début des signes (en mois)",
        "Âge de découverte de la drépanocytose (en mois)",
        "Nbre d'hospitalisations avant 2017",
        "Nbre d'hospitalisations entre 2017 et 2023",
        "Nbre de transfusion avant 2017",
        "Nbre de transfusion Entre 2017 et 2023",
        "HDJ",
        "Sexe", "Origine Géographique", "Parents Salariés",
        "PEV Complet", "Vaccin contre pneumocoque", "Vaccin contre méningocoque", 
        "Vaccin contre Les salmonelles", "L'hydroxyurée", "Echange transfusionnelle",  
        "Prise en charge", "Prophylaxie à la pénicilline", "CVO", "Anémie", "AVC", 
        "STA", "Priapisme", "Infections", "Ictère", "Type de drépanocytose",
    ]
    df_selected = df[variables_selected].copy()

    # ================================
    # 3. Encodage des variables qualitatives
    # ================================
    binary_mappings = {
        "Sexe": {"Masculin": 1, "Féminin": 0, "M": 1, "F": 0},
        "Parents Salariés": {"OUI": 1, "NON": 0},
        "PEV Complet": {"OUI": 1, "NON": 0},
        "Vaccin contre pneumocoque": {"OUI": 1, "NON": 0},
        "Vaccin contre méningocoque": {"OUI": 1, "NON": 0},
        "Vaccin contre Les salmonelles": {"OUI": 1, "NON": 0},
        "L'hydroxyurée": {"OUI": 1, "NON": 0},
        "Echange transfusionnelle": {"OUI": 1, "NON": 0},
        "Prophylaxie à la pénicilline": {"OUI": 1, "NON": 0},
        "CVO": {"OUI": 1, "NON": 0},
        "Anémie": {"OUI": 1, "NON": 0},
        "AVC": {"OUI": 1, "NON": 0},
        "STA": {"OUI": 1, "NON": 0},
        "Priapisme": {"OUI": 1, "NON": 0},
        "Infections": {"OUI": 1, "NON": 0},
        "Ictère": {"OUI": 1, "NON": 0},
    }

    df_selected.replace(binary_mappings, inplace=True)
    df_selected = pd.get_dummies(df_selected, columns=["Origine Géographique"], drop_first=False)
    df_selected = pd.get_dummies(df_selected, columns=["Prise en charge"], drop_first=True)
    df_selected = pd.get_dummies(df_selected, columns=["Type de drépanocytose"], drop_first=False)

    # ================================
    # 4. Standardisation
    # ================================
    quantitative_vars = [
        "Âge du debut d etude en mois (en janvier 2023)",
        "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
        "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)",
        "Âge de début des signes (en mois)",
        "Âge de découverte de la drépanocytose (en mois)",
        "Nbre d'hospitalisations avant 2017",
        "Nbre d'hospitalisations entre 2017 et 2023",
        "Nbre de transfusion avant 2017",
        "Nbre de transfusion Entre 2017 et 2023",
        "HDJ"
    ]
    scaler = StandardScaler()
    df_scaled = df_selected.copy()
    df_scaled[quantitative_vars] = scaler.fit_transform(df_selected[quantitative_vars])

    # ================================
    # 5. Onglets horizontaux
    # ================================
    tabs = st.tabs(["Vue d'ensemble", "Visualisation ACP", "Profil détaillé"])

    # --- Onglet 1 : Vue d'ensemble ---
    with tabs[0]:
        st.subheader("Méthodologie et Graphe du coude")
        st.markdown("""
        ### Étapes méthodologiques utilisées :
        1. **Prétraitement des données** : nettoyage et sélection des variables pertinentes.  
        2. **Encodage** : transformation des variables qualitatives (binaire et catégorielle).  
        3. **Standardisation** : mise à l’échelle des variables quantitatives en z-scores.  
        4. **Clustering KMeans** : segmentation des patients en groupes homogènes.  
        5. **Analyse en Composantes Principales (ACP)** : visualisation 2D des clusters.  
        """)

        inertia = []
        K_range = range(1, 11)
        for k in K_range:
            kmeans_test = KMeans(n_clusters=k, random_state=42)
            kmeans_test.fit(df_scaled)
            inertia.append(kmeans_test.inertia_)

        fig, ax = plt.subplots()
        ax.plot(list(K_range), inertia, marker='o')
        ax.set_xlabel("Nombre de clusters (k)")
        ax.set_ylabel("Inertia (SSE)")
        ax.set_title("Graphe du coude pour KMeans")
        st.pyplot(fig)

        n_clusters = st.slider("Sélectionner le nombre de clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df_scaled)
        df_scaled["Cluster"] = df["Cluster"]
        st.success("Clustering effectué !")

    # --- Onglet 2 : Visualisation ACP ---
    with tabs[1]:
        st.subheader("Visualisation des clusters via PCA 2D")
        pca = PCA(n_components=2)
        components = pca.fit_transform(df_scaled.drop("Cluster", axis=1))
        df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
        df_pca['Cluster'] = df['Cluster']

        # Variance expliquée
        explained_var = pca.explained_variance_ratio_
        st.write(f"**Variance expliquée par PC1 :** {explained_var[0]:.2%}")
        st.write(f"**Variance expliquée par PC2 :** {explained_var[1]:.2%}")
        st.write(f"**Variance totale expliquée par PC1 et PC2 :** {(explained_var[0]+explained_var[1]):.2%}")

        fig, ax = plt.subplots()
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
        ax.set_title("Clusters visualisés sur les 2 premières composantes principales")
        st.pyplot(fig)

    # --- Onglet 3 : Profil détaillé ---
    with tabs[2]:
        st.subheader("Profil détaillé des clusters")

        # Nombre de patients par cluster
        st.write("### Nombre de patients par cluster")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        st.dataframe(cluster_counts.rename("Nombre de patients"))

        # Données complètes avec attribution de cluster (variables sélectionnées par utilisateur)
        st.write("### Données complètes avec attribution de cluster")
        cols_to_show = st.multiselect(
            "Sélectionnez les variables à afficher :",
            options=variables_selected,
            default=quantitative_vars[:3]  # par défaut, afficher les 3 premières variables quantitatives
        )
        if cols_to_show:
            st.dataframe(df[cols_to_show + ["Cluster"]])
        else:
            st.warning("Veuillez sélectionner au moins une variable à afficher.")

        # Moyennes Z-scores par cluster
        st.write("### Moyennes standardisées (Z-scores) des variables par cluster :")
        cluster_means = df_scaled.groupby("Cluster").mean().T

        interpretations = []
        for var, row in cluster_means.iterrows():
            max_cluster = row.idxmax()
            min_cluster = row.idxmin()
            if abs(row[max_cluster]) < 0.2 and abs(row[min_cluster]) < 0.2:
                interp = "Peu discriminant"
            else:
                interp = f"Plus élevé dans Cluster {max_cluster}, plus bas dans Cluster {min_cluster}"
            interpretations.append(interp)
        cluster_means["Interprétation"] = interpretations
        st.dataframe(cluster_means)


