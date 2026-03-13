import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import os

st.set_page_config(page_title="MediPredict", layout="centered")

# --- Chargement modèle ---
@st.cache_resource
def load_model():
    model = joblib.load("medipredict_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

model, scaler = load_model()
df = load_data()
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# --- Consentement ---
if "consent" not in st.session_state:
    st.session_state.consent = False

if not st.session_state.consent:
    st.title("MediPredict")
    st.write("Outil de sensibilisation au risque de diabète de type 2 — SantéCo")
    st.warning("**Mention légale** : Cet outil est un outil de sensibilisation. Il ne constitue pas un avis médical. En cas de doute, consultez un professionnel de santé.")
    st.write("""
**Politique de confidentialité** : Aucune donnée saisie n'est stockée ni transmise.
Les informations sont traitées uniquement en mémoire le temps de votre session
et supprimées dès la fermeture de la page. Aucun tiers n'y a accès.
""")
    if st.button("J'ai compris et je souhaite continuer"):
        st.session_state.consent = True
        st.rerun()
    st.stop()

# --- Navigation ---
page = st.sidebar.radio("Navigation", ["Accueil", "Mon profil de risque", "Comprendre ma prédiction", "Explorer les données"])

# =====================
# PAGE 1 — ACCUEIL
# =====================
if page == "Accueil":
    st.title("MediPredict")
    st.write("Bienvenue sur MediPredict, un outil de sensibilisation proposé par SantéCo.")
    st.write("""
Cet outil vous permet de renseigner quelques indicateurs de santé et d'obtenir
une estimation de votre niveau de risque de développer un diabète de type 2.

**Ce que cet outil fait :**
- Estimer un niveau de risque (faible, modéré, élevé)
- Vous expliquer quels facteurs ont influencé ce résultat
- Vous orienter vers un professionnel de santé si nécessaire

**Ce que cet outil ne fait pas :**
- Poser un diagnostic médical
- Remplacer une consultation
- Stocker vos données
""")
    st.info("Cet outil est un outil de sensibilisation. Il ne constitue pas un avis médical. En cas de doute, consultez un professionnel de santé.")
    st.write("Pour commencer, rendez-vous dans **Mon profil de risque** dans le menu à gauche.")

# =====================
# PAGE 2 — PROFIL
# =====================
elif page == "Mon profil de risque":
    st.title("Mon profil de risque")
    st.write("Renseignez vos indicateurs de santé ci-dessous.")

    with st.form("profil_form"):
        pregnancies_na = st.checkbox("La variable 'Nombre de grossesses' ne me concerne pas")
        if pregnancies_na:
            pregnancies = 0  # médiane dataset
        else:
            pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=1, help="Nombre de grossesses antérieures")

        glucose = st.number_input("Taux de glucose (mg/dL)", min_value=40, max_value=400, value=120,
            help="Concentration de glucose plasmatique. Valeur normale à jeun : 70-100 mg/dL")
        blood_pressure = st.number_input("Pression artérielle diastolique (mm Hg)", min_value=20, max_value=150, value=72,
            help="Pression lors du relâchement du cœur. Valeur normale : 60-80 mm Hg")
        skin_thickness = st.number_input("Épaisseur du pli cutané du triceps (mm)", min_value=5, max_value=100, value=23,
            help="Mesure de la graisse sous-cutanée. Valeur normale : 10-40 mm")
        insulin = st.number_input("Insuline sérique à 2h (µU/ml)", min_value=10, max_value=900, value=80,
            help="Taux d'insuline dans le sang. Valeur normale : 16-166 µU/ml")
        bmi = st.number_input("Indice de masse corporelle (kg/m²)", min_value=10.0, max_value=80.0, value=32.0, step=0.1,
            help="Poids / taille². Normal : 18.5-24.9. Surpoids : 25-29.9. Obésité : > 30")
        dpf = st.number_input("Score d'antécédents familiaux de diabète", min_value=0.05, max_value=2.5, value=0.47, step=0.01,
            help="Score calculé selon les antécédents familiaux. Plus il est élevé, plus le risque génétique est important.")
        age = st.number_input("Âge (années)", min_value=21, max_value=100, value=33,
            help="Votre âge en années")

        submitted = st.form_submit_button("Analyser mon profil")

    if submitted:
        input_dict = {
            'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
            'DiabetesPedigreeFunction': dpf, 'Age': age
        }
        x = np.array([[input_dict[f] for f in feature_names]])
        x_scaled = scaler.transform(x)
        proba = model.predict_proba(x_scaled)[0][1]

        if proba < 0.35:
            niveau = "faible"
            couleur = "green"
        elif proba < 0.65:
            niveau = "modéré"
            couleur = "orange"
        else:
            niveau = "élevé"
            couleur = "red"

        st.session_state["input_dict"] = input_dict
        st.session_state["x_scaled"] = x_scaled
        st.session_state["proba"] = proba
        st.session_state["niveau"] = niveau

        st.markdown("---")
        st.subheader("Résultat")
        st.markdown(f"**Niveau de risque estimé : {niveau.upper()}**")

        # Jauge simple sans couleur seule
        barre = int(proba * 20)
        st.progress(proba)
        st.write(f"Score interne : {barre}/20 — Risque **{niveau}**")

        if niveau == "élevé":
            st.error("Risque élevé détecté. Consultez un médecin ou un professionnel de santé.")
        elif niveau == "modéré":
            st.warning("Risque modéré. Une consultation préventive est recommandée.")
        else:
            st.success("Risque faible. Continuez à adopter de bonnes habitudes de vie.")

        st.write("Rendez-vous dans **Comprendre ma prédiction** pour les détails.")
        st.info("Cet outil est un outil de sensibilisation. Il ne constitue pas un avis médical.")

# =====================
# PAGE 3 — EXPLICATION
# =====================
elif page == "Comprendre ma prédiction":
    st.title("Comprendre ma prédiction")

    if "x_scaled" not in st.session_state:
        st.warning("Veuillez d'abord renseigner votre profil dans 'Mon profil de risque'.")
        st.stop()

    x_scaled = st.session_state["x_scaled"]
    input_dict = st.session_state["input_dict"]
    niveau = st.session_state["niveau"]

    st.write(f"Votre niveau de risque estimé est : **{niveau}**")

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame(x_scaled, columns=feature_names))
    vals = shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[1][0]

    st.subheader("Facteurs ayant influencé ce résultat")

    # Graphique SHAP simple
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['tomato' if v > 0 else 'steelblue' for v in vals]
    ax.barh(feature_names, vals, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Contribution au risque (rouge = augmente, bleu = diminue)")
    ax.set_title("Impact de chaque variable sur votre résultat")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Explication en langage naturel
    top_idx = int(np.argmax(np.abs(vals)))
    top_feature = feature_names[top_idx]
    direction = "augmenter" if vals[top_idx] > 0 else "diminuer"

    labels_fr = {
        'Glucose': 'votre taux de glucose',
        'BMI': 'votre indice de masse corporelle',
        'Age': 'votre âge',
        'DiabetesPedigreeFunction': 'vos antécédents familiaux',
        'Insulin': 'votre taux d\'insuline',
        'BloodPressure': 'votre pression artérielle',
        'SkinThickness': 'l\'épaisseur de votre pli cutané',
        'Pregnancies': 'le nombre de grossesses'
    }

    st.subheader("En résumé")
    st.write(f"Le facteur ayant le plus influencé votre résultat est **{labels_fr[top_feature]}**, "
             f"qui a tendance à **{direction}** votre niveau de risque estimé.")

    # Recommandations
    st.subheader("Recommandations génériques")
    if input_dict['Glucose'] > 140:
        st.write("- Un taux de glucose élevé est un facteur de risque modifiable. Des ajustements alimentaires peuvent aider.")
    if input_dict['BMI'] > 30:
        st.write("- Un IMC supérieur à 30 est associé à un risque accru. Une activité physique régulière est recommandée.")
    if input_dict['Age'] > 45:
        st.write("- Le risque de diabète de type 2 augmente avec l'âge. Un suivi médical régulier est conseillé.")
    st.write("- En cas de résultat préoccupant, consultez un médecin. [Trouver un médecin sur ameli.fr](https://www.ameli.fr)")

    # Comparaison profil vs dataset
    st.subheader("Votre profil comparé au dataset")
    selected_var = st.selectbox("Choisir une variable à comparer", feature_names)
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.hist(df[selected_var], bins=20, color='lightgray', edgecolor='white', label='Dataset')
    ax2.axvline(input_dict[selected_var], color='tomato', linewidth=2, label='Votre valeur')
    ax2.set_xlabel(selected_var)
    ax2.set_ylabel("Fréquence")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.info("Cet outil est un outil de sensibilisation. Il ne constitue pas un avis médical.")

# =====================
# PAGE 4 — DONNÉES
# =====================
elif page == "Explorer les données":
    st.title("Explorer les données")

    st.subheader("Distribution des variables")
    var = st.selectbox("Variable", feature_names)
    fig, ax = plt.subplots(figsize=(7, 3))
    df[var].hist(ax=ax, bins=20, color='steelblue', edgecolor='white')
    ax.set_title(f"Distribution — {var}")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Corrélations")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    import seaborn as sns
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2, annot_kws={"size": 8})
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.subheader("Performance du modèle")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    import seaborn as sns

    df_clean = df.copy()
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df_clean[col] = df_clean[col].replace(0, df_clean[col].median())

    X = df_clean[feature_names]
    y = df_clean['Outcome']
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Matrice de confusion**")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_xlabel("Prédit")
        ax3.set_ylabel("Réel")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col2:
        st.write("**Courbe ROC**")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        ax4.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax4.plot([0, 1], [0, 1], 'k--')
        ax4.set_xlabel("Taux faux positifs")
        ax4.set_ylabel("Taux vrais positifs")
        ax4.legend()
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.subheader("Transparence et limites du modèle")
    st.write("""
**Modèle utilisé** : Random Forest (100 arbres, random_state=42)

**Performances sur le jeu de test** :
- Accuracy : 74.7%
- F1-Score : 62.9%
- AUC-ROC : 0.820

**Limites identifiées** :
- Le dataset Pima Indians est composé exclusivement de femmes amérindiennes âgées de 21 ans minimum. Les prédictions peuvent être moins fiables pour des hommes ou des personnes d'autres origines.
- L'insuline avait 48.7% de valeurs manquantes, imputées par la médiane — ce qui réduit la précision sur cette variable.
- Le rappel de 61% signifie que 39% des cas diabétiques ne sont pas détectés.

**Ce modèle ne doit pas être utilisé comme outil de diagnostic médical.**
""")
