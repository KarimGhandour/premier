# import des librairies nécessaires 
# ici streamlit est utilisée pour créer une application web interactive
import streamlit as st
from datetime import datetime
# on importe également les fonctions crées au sein du notebook précédent
from notebook_functions import predire_matchs, equipes

# configuration initiale de la page
st.set_page_config(page_title='PremierPredict', page_icon='⚽️', layout='wide', initial_sidebar_state='expanded')

# titre principal de l'application et noms des membres de l'équipe et ajout de détails sur le projet
st.markdown('## ⚽️ PremierPredict : *Prédictions des matchs de la Premier League*')
url = 'https://ghj95.github.io/portfolio//'
st.markdown(
    f"<a href='{url}' target='_blank' style='text-decoration: none; color: inherit;'>`Par : Gabriel Hardy-Joseph, Karim Ghandour, Louis-Matteo Creplet et Mohamed-Aziz Charfeddine`</a>",
    unsafe_allow_html=True,
)
st.markdown(
    "<span style='text-decoration: none;'><code style='color: inherit;'>Projet réalisé dans le cadre du cours TECH20704</code></span>",
    unsafe_allow_html=True,
)

# définition d'une fonction pour afficher les informations sur le fonctionnement de l'application
def appinfo():
    # premier conteneur expliquant le modèle et ses données d'entraînement avec st.container
    with st.container(border=True):
        st.write("Cette application s'appuie sur un modèle de classification **Extreme Gradient Boosting (XGBoost)** entraîné sur 25 ans de résultats historiques (2000-2025). Les prédictions intègrent la *forme récente des équipes*, l'*analyse statistique des matchs précédents* et l'*historique des confrontations directes* entre les clubs. Essayez-la!")
    # second conteneur avec un avertissement sur la fiabilité des prédictions
    with st.container(border=True):
        st.write('Les prédictions du modèle ne sont pas des certitudes.')

# ajout d'un titre dans la barre latérale pour les informations à saisir avec la méthode st.sidebar
st.sidebar.header('Détails du match')

# définition d'une fonction pour recueillir les entrées de l'utilisateur
def input_features():
    # sélection de l'équipe jouant à domicile via une liste déroulante avec la méthode .selectbox
    domicile = st.sidebar.selectbox('Équipe à domicile :', (equipes), index=None, placeholder='Séléctionez une équipe...')
    # sélection de l'équipe jouant à l'extérieur via une autre liste déroulante
    exterieur = st.sidebar.selectbox("Équipe à l'extérieur :", (equipes), index=None, placeholder='Séléctionez une équipe...')
    # sélection de la date du match avec la date du jour comme valeur par défaut avec la méthode .date_input
    match_date = st.sidebar.date_input('Date du match :', datetime.today())
    return domicile, exterieur, match_date

# récupération des choix de l'utilisateur via la fonction précédente par unpacking
domicile, exterieur, match_date = input_features()

# appel de la fonction pour afficher les informations sur l'application
appinfo()
'---'
    
# on initialise la variable qui contiendra la prédiction
prediction = None
# ajout d'un bouton dans la barre latérale pour déclencher la prédiction
if st.sidebar.button('Prédire'):
    # on crée maintenant une liste de conditions pour s'assurer que les entrées des utilisateurs sont valides avec une série de vérifications
    # on assume par défaut que les entrées sont valides
    valid_input = True
    # vérification que les équipes sélectionnées font partie de la liste des équipes disponibles
    if domicile and exterieur not in equipes:
        st.error('Veuillez séléctionner deux équipes parmi celles dans la liste')
        valid_input = False
    # vérification que l'utilisateur n'a pas sélectionné la même équipe deux fois (domicile et extérieur)
    if domicile == exterieur:
        st.error('Veuillez sélectionner deux équipes différentes')
        valid_input = False
    # vérification que la date n'est pas trop éloignée dans le futur pour garantir la pertinence de la prédiction
    if match_date.year > datetime.today().year + 1:
        st.error('Veuillez sélectionner une date moins éloignée')
        valid_input = False
    # vérification que la date n'est pas trop ancienne pour garantir la pertinence de la prédiction
    if match_date.year < datetime.today().year - 1:
        st.error('Veuillez sélectionner une date moins éloignée')
        valid_input = False
    
    # si toutes les vérifications sont passées avec succès, on effectue la prédiction
    if valid_input:
        # affichage d'un indicateur de chargement pendant le calcul de la prédiction avec la méthode st.spinner
        with st.spinner('Chargement des prédictions...'):
            # appel de la fonction de prédiction avec les paramètres fournis par l'utilisateur
            prediction, prob = predire_matchs(domicile, exterieur, match_date)

    # affichage du résultat de la prédiction avec formatage pour mettre en évidence l'équipe gagnante et le pourcentage de confiance
    if prediction == 0:
        # cas d'une victoire prédite pour l'équipe extérieure
        st.success(f'Le modèle prédit une victoire de **{exterieur}** à **{prob[0, 0]:.0%}**!', icon='🎯')
    elif prediction == 1:
        # cas d'une victoire prédite pour l'équipe à domicile
        st.success(f'Le modèle prédit une victoire de **{domicile}** **{prob[0, 1]:.0%}**!', icon='🎯')
    else:
        # cas d'un match nul prédit
        st.success(f'Le modèle prédit un match nul à **{prob[0, 2]:.0%}**!', icon='🎯')