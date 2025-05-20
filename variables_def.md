# Variables explicatives du modèle PremierPredict

## Codes d'équipes
- **ExterieurCode** : Code numérique identifiant l'équipe jouant à l'extérieur
- **DomicileCode** : Code numérique identifiant l'équipe jouant à domicile

## Statistiques de buts
- **DiffButsGlobal** : Différence entre les différentiels de buts globaux des deux équipes
- **DiffButs** : Différence entre le différentiel de buts à domicile de l'équipe hôte et celui à l'extérieur de l'équipe visiteuse
- **DiffButsExterieur** : Différence moyenne entre les buts marqués et encaissés par l'équipe extérieure lorsqu'elle joue à l'extérieur
- **DiffButsDomicile** : Différence moyenne entre les buts marqués et encaissés par l'équipe à domicile lorsqu'elle joue chez elle
- **DomicileAvgButsMarques_Home** : Moyenne de buts marqués par l'équipe à domicile lors de ses matchs à domicile
- **DomicileAvgButsMarques_Away** : Moyenne de buts marqués par l'équipe à domicile lors de ses matchs à l'extérieur
- **DomicileAvgButsEncaisses_Home** : Moyenne de buts encaissés par l'équipe à domicile lors de ses matchs à domicile
- **DomicileAvgButsEncaisses_Away** : Moyenne de buts encaissés par l'équipe à domicile lors de ses matchs à l'extérieur
- **ExterieurAvgButsMarques_Home** : Moyenne de buts marqués par l'équipe extérieure lors de ses matchs à domicile
- **ExterieurAvgButsMarques_Away** : Moyenne de buts marqués par l'équipe extérieure lors de ses matchs à l'extérieur
- **ExterieurAvgButsEncaisses_Home** : Moyenne de buts encaissés par l'équipe extérieure lors de ses matchs à domicile
- **ExterieurAvgButsEncaisses_Away** : Moyenne de buts encaissés par l'équipe extérieure lors de ses matchs à l'extérieur

## Forme des équipes
- **DomicileForme** : Points accumulés par l'équipe à domicile lors de ses 5 derniers matchs (3 pour victoire, 1 pour nul)
- **ExterieurForme** : Points accumulés par l'équipe extérieure lors de ses 5 derniers matchs
- **DiffForme** : Différence entre les points de forme des deux équipes

## Historique face-à-face (FAF)
- **FAF_Diff** : Différence entre le nombre de victoires de l'équipe à domicile à domicile et de l'équipe extérieure à l'extérieur
- **FAF_DiffGlobal** : Différence globale de victoires entre les deux équipes lors de leurs confrontations
- **FAF_DiffNul** : Différence entre le nombre de matchs nuls lorsque l'équipe domicile joue à domicile et à l'extérieur
- **FAF_Nul_Domicile_Dom** : Nombre de matchs nuls lorsque l'équipe domicile joue à domicile face à l'équipe extérieure
- **FAF_Nul_Domicile_Ext** : Nombre de matchs nuls lorsque l'équipe domicile joue à l'extérieur face à l'équipe extérieure
- **FAF_VictoiresDomicile_Dom** : Nombre de victoires de l'équipe à domicile à domicile contre l'équipe extérieure
- **FAF_VictoiresDomicile_Ext** : Nombre de victoires de l'équipe à domicile lorsqu'elle joue à l'extérieur contre l'équipe extérieure
- **FAF_VictoiresExterieur_Dom** : Nombre de victoires de l'équipe extérieure lorsqu'elle joue à domicile contre l'équipe à domicile
- **FAF_VictoiresExterieur_Ext** : Nombre de victoires de l'équipe extérieure à l'extérieur contre l'équipe à domicile

## Variables temporelles
- **AnneeSaison** : Année de début de la saison en cours
- **Mois** : Mois du match
- **SemaineSaison** : Semaine de la saison (à partir du 5 août) dans laquelle le match a lieu