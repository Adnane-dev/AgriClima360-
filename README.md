# 🌦️ AgriClima360 
### Pipeline CRISP-DM & Visualisation Massive des Données Climatiques pour l'Agriculture de Précision

**Interface web :** [https://agriclima360-f.streamlit.app/](https://agriclima360-f.streamlit.app/)

**Auteurs :**
- 👨‍💻 [Adnane Mahamadou Saadou](https://github.com/Adnane-dev)
- 👩‍💻 [Radhia Darghoothi](https://github.com/RadhiaDarghoothi)

**Repository :** [https://github.com/Adnane-dev/AgriClima360-](https://github.com/Adnane-dev/AgriClima360-)

---

## 📖 Description

**AgriClima360** est une plateforme complète d'analyse prédictive des données climatiques pour l'agriculture de précision. Ce projet implémente un pipeline CRISP-DM complet avec visualisation massive des données climatiques NOAA GHCN couvrant la période 2000-2024.

## 🎯 Objectifs Principaux

- ✅ **Pipeline CRISP-DM** : Implémentation complète des 6 phases méthodologiques
- 🔄 **Machine Learning** : Modèles prédictifs optimisés pour l'agriculture
- 📊 **Visualisation Massive** : Analyse de grands volumes de données climatiques
- 🌐 **Dashboard Interactif** : Interface Streamlit pour l'exploration en temps réel
- 🚀 **Analyse d'Impact** : Évaluation des impacts climatiques sur les rendements agricoles

## 🏗️ Architecture du Projet

```
AgriClima360/
├── app/                           # 📱 Application Streamlit
│   └── streamlit_app.py
├── data/                          # 🗃️ Données structurées
│   ├── raw/                       # Données brutes NOAA
│   ├── processed/                 # Données traitées
│   └── models/                    # Modèles entraînés
├── notebooks/                     # 🔬 Analyses exploratoires
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   └── 03_modeling.ipynb
├── visualisation/                 # 📈 Modules de visualisation
│   ├── temporal_analysis.py
│   ├── spatial_maps.py
│   └── agricultural_impact.py
├── src/                           # 💻 Code source
│   ├── data_processing/
│   ├── modeling/
│   └── utils/
├── docs/                          # 📄 Documentation
├── requirements.txt               # Dépendances Python
└── README.md                      # Ce fichier
```

## 🚀 Démarrage Rapide

### Prérequis

- Python 3.9 ou supérieur
- pip (gestionnaire de packages Python)
- Git

### Installation

```bash
# Cloner le repository
git clone https://github.com/Adnane-dev/AgriClima360-.git
cd AgriClima360-

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Utilisation

```bash
# Lancer le dashboard principal
streamlit run app/streamlit_app.py

# Explorer les notebooks d'analyse
jupyter notebook notebooks/

# Exécuter l'analyse des modèles
python src/modeling/model_analysis.py
```

## 📊 Fonctionnalités

### 🔍 Pipeline CRISP-DM Complet

1. **Compréhension Métier** : Définition des objectifs et enjeux agricoles
2. **Exploration des Données** : Analyse approfondie des données climatiques NOAA GHCN
3. **Préparation des Données** : Nettoyage, transformation et feature engineering
4. **Modélisation** : Développement d'algorithmes de Machine Learning
5. **Évaluation** : Validation rigoureuse des performances des modèles
6. **Déploiement** : Mise en production via interface Streamlit

### 📈 Visualisations Avancées

- **Analyses Temporelles** : Évolution des tendances climatiques de 2000 à 2024
- **Cartographies Interactives** : Répartition géographique des impacts climatiques
- **Graphiques Dynamiques** : Exploration interactive des données multidimensionnelles
- **Dashboard Unifié** : Vue d'ensemble consolidée des indicateurs clés de performance

### 🌱 Analyse d'Impact Agricole

- **Stress Hydrique** : Identification et quantification des risques de sécheresse
- **Extrêmes Climatiques** : Analyse de l'impact des températures critiques sur les cultures
- **Rendements Agricoles** : Corrélations entre variables climatiques et productivité
- **Recommandations Stratégiques** : Préconisations pour l'adaptation climatique

## 📁 Structure des Données

### Sources Principales

- **NOAA GHCN** (Global Historical Climatology Network) : Données climatiques historiques mondiales
- **Données Agricoles** : Indicateurs de rendements et pratiques culturales
- **Période de couverture** : 2000-2024 (25 ans)
- **Variables climatiques** : Température (min/max/moy), précipitations, humidité, vitesse du vent

### Métriques Agricoles Calculées

- Indices de stress hydrique (Water Stress Index)
- Périodes de croissance optimales par culture
- Scores de risque climatique multi-factoriels
- Indicateurs de résilience et d'adaptation

## 🔧 Technologies Utilisées

### Stack Data Science

```python
# Traitement et analyse des données
pandas>=2.0.0          # Manipulation de données
numpy>=1.24.0          # Calculs numériques
scikit-learn>=1.3.0    # Machine Learning

# Visualisation  
matplotlib>=3.7.0      # Graphiques statiques
seaborn>=0.12.0        # Visualisations statistiques
plotly>=5.15.0         # Graphiques interactifs
folium>=0.14.0         # Cartographie

# Application web
streamlit>=1.28.0      # Dashboard interactif
altair>=5.0.0          # Grammaire de visualisation

# Environnement de développement
jupyter>=1.0.0         # Notebooks interactifs
```

### Analyse Spatiale

- **Cartes interactives** : Folium, Plotly Express
- **Géolocalisation** : Intégration des coordonnées des stations météo NOAA
- **Zonage climatique** : Clustering géographique et classification spatiale

## 📈 Résultats et Insights Clés

### Tendances Climatiques Observées

- Augmentation progressive des températures moyennes annuelles
- Variabilité accrue des régimes de précipitations
- Fréquence croissante des événements climatiques extrêmes
- Décalage des saisons agricoles optimales

### Impacts sur l'Agriculture

- Corrélations significatives entre anomalies climatiques et variations de rendement
- Identification de zones géographiques à risque élevé
- Détermination de périodes critiques pour les principales cultures
- Opportunités d'optimisation des pratiques culturales

## 👥 Équipe de Développement

| Membre | Rôle Principal | Contributions Spécifiques |
|--------|----------------|---------------------------|
| **👨‍💻 Adnane Mahamadou Saadou** | Data Engineering & ML | Pipeline de données, modélisation prédictive, architecture système |
| **👩‍💻 Radhia Darghoothi** | Data Visualization & UX | Dashboard Streamlit, visualisations interactives, reporting |

## 🤝 Contribution au Projet

Les contributions sont les bienvenues ! Pour contribuer :

1. **Fork** le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/NouvelleFonctionnalite`)
3. Commiter vos modifications (`git commit -m 'Ajout d'une nouvelle fonctionnalité'`)
4. Pousser vers la branche (`git push origin feature/NouvelleFonctionnalite`)
5. Ouvrir une **Pull Request** avec une description détaillée

### Guidelines de Contribution

- Respecter le style de code existant (PEP 8 pour Python)
- Ajouter des tests unitaires pour les nouvelles fonctionnalités
- Mettre à jour la documentation en conséquence
- Décrire clairement les changements dans la Pull Request

## 📄 Licence

Ce projet est distribué sous licence **MIT**. Consultez le fichier [LICENSE](LICENSE) pour plus de détails.

## 📞 Contact

- **Adnane Mahamadou Saadou** - [GitHub](https://github.com/Adnane-dev)
- **Radhia Darghoothi** - [GitHub](https://github.com/RadhiaDarghoothi)

Pour toute question ou suggestion, n'hésitez pas à ouvrir une **issue** sur GitHub.

---

## 🚀 Déploiement Continu

### Configuration GitHub Actions

Le fichier `.github/workflows/deploy.yml` configure le déploiement automatique :

```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/ --verbose
      
      - name: Deploy to Streamlit Cloud
        run: |
          streamlit run app/streamlit_app.py --server.port 8501 &
```

---

**🌱 Développé pour une agriculture résiliente face aux changements climatiques**

*Projet académique - Ingénierie des Données & Visualisation Massive - 2024*

---

## 📚 Références

- [NOAA Global Historical Climatology Network](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)
- [Méthodologie CRISP-DM](https://www.datascience-pm.com/crisp-dm-2/)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
