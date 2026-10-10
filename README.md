# building-perceptron

Projet pédagogique d'implémentation d'un perceptron pour la classification de tumeurs du sein.

## Contexte du projet

Ce projet s'inscrit dans le cadre de l'apprentissage de l'intelligence artificielle et du machine learning. L'objectif est de :

1. **Comprendre le fonctionnement fondamental d'un perceptron** à travers une implémentation simple et pédagogique
2. **Développer un perceptron compatible avec Scikit-learn** pour résoudre un problème de classification médicale
3. **Réaliser une analyse exploratoire complète** (EDA) des données 
4. **Évaluer et comparer les performances** du modèle personnalisé


## Données

### Source

Le projet utilise le **Breast Cancer Wisconsin (Diagnostic) Dataset** pour classifier des tumeurs comme bénignes ou malignes.

**Dataset** (`raw_data/bcw_data.csv`)
https://drive.google.com/file/d/1itXdRo4WJuhqCjtVX4WGvT327WWp4LB7/view

### Description
- **Nombre de caractéristiques** : 30 features numériques calculées à partir d'images de masses cellulaires
- **Variable cible** : `diagnosis` (M = Maligne, B = Bénigne)
- **Types de features** :
  - Mesures moyennes (_mean) : rayon, texture, périmètre, aire, etc.
  - Erreurs standard (_se) : variabilité des mesures
  - Valeurs maximales (_worst) : cas les plus extrêmes

### Caractéristiques principales
```
- radius_mean : rayon moyen des cellules
- texture_mean : écart-type des valeurs de niveau de gris
- perimeter_mean : périmètre moyen
- area_mean : aire moyenne
- smoothness_mean : variation locale des longueurs de rayon
- compactness_mean : (périmètre² / aire) - 1.0
- concavity_mean : sévérité des portions concaves du contour
- concave points_mean : nombre de portions concaves du contour
- symmetry_mean : symétrie
- fractal_dimension_mean : "approximation de côte" - 1
```

## Structure du projet

```
├── app.py
├── data
│   ├── processed_data
│   │   ├── bcw_test.csv
│   │   └── bcw_train.csv
│   └── raw_data
│       └── bcw_data.csv
├── documentation
│   └── Questions.pptx
├── notebooks
│   ├── archives
│   │   ├── create-db-incendies.ipynb
│   │   ├── eda_archive.ipynb
│   │   └── eda_utils.py
│   ├── eda.ipynb
│   └── training.ipynb
├── pyproject.toml
├── README.md
├── scripts
│   └── eda
│       └── run_eda.py
├── src
│   └── building_perceptron
│       ├── data
│       │   ├── data_utils.py
│       │   └── __init__.py
│       ├── __init__.py
│       ├── model
│       │   ├── __init__.py
│       │   ├── logistic_regression.py
│       │   └── perceptron.py
│       ├── services
│       │   └── __init__.py
│       └── views
│           └── __init__.py
└── uv.lock


## Analyse

### Processus d'analyse exploratoire (EDA)

Le notebook `eda.ipynb` réalise une analyse complète des données :
- Chargement et inspection initiale du dataset
- Détection et traitement des valeurs manquantes
- Identification et gestion des valeurs aberrantes
- Analyse de la distribution des variables
- Visualisation des corrélations entre features
- Équilibrage des classes (Maligne vs Bénigne)

