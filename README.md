# 🔬 Prédiction de Maladie Cardiaque avec Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

---

## 📌 Description du Projet

Ce projet utilise le **Machine Learning** pour prédire la présence d'une maladie cardiaque chez un patient à partir de données cliniques. 

L'objectif est de développer un modèle **fiable**, **interprétable** et **performant** pour aider au diagnostic précoce des maladies cardiovasculaires, qui sont la **première cause de mortalité dans le monde** (17,9 millions de décès par an selon l'OMS).

### 🎯 Problématique
> **"Comment prédire la présence d'une maladie cardiaque à partir des données cliniques d'un patient ?"**

### 🏆 Résultats Clés
- **Meilleur modèle** : Logistic Regression
- **Accuracy** : 89.13%
- **F1-Score** : 90.38%
- **Recall** : 92.16% *(détecte 92 malades sur 100)*
- **ROC-AUC** : 93.33%

---

## 📊 Dataset

### Source
- **Plateforme** : [Kaggle](https://www.kaggle.com/)
- **Dataset** : [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Utilisé par** : +500 chercheurs dans le monde

### Caractéristiques

| Caractéristique | Valeur |
|-----------------|--------|
| **Patients** | 918 |
| **Variables** | 12 |
| **Variables catégorielles** | 5 |
| **Variables numériques** | 6 |
| **Variable cible** | HeartDisease (0=Sain, 1=Malade) |
| **Distribution** | 55% malades / 45% sains |

### Variables

| Variable | Description | Type |
|----------|-------------|------|
| **Age** | Âge du patient (années) | Numérique |
| **Sex** | M (Masculin) / F (Féminin) | Catégorielle |
| **ChestPainType** | Type de douleur thoracique (ATA, ASY, NAP, TA) | Catégorielle |
| **RestingBP** | Pression artérielle au repos (mm Hg) | Numérique |
| **Cholesterol** | Taux de cholestérol sérique (mg/dl) | Numérique |
| **FastingBS** | Glycémie à jeun > 120 mg/dl (0/1) | Binaire |
| **RestingECG** | Résultat ECG au repos (Normal, ST, LVH) | Catégorielle |
| **MaxHR** | Fréquence cardiaque maximale | Numérique |
| **ExerciseAngina** | Angine d'effort (Y/N) | Binaire |
| **Oldpeak** | Dépression du segment ST | Numérique |
| **ST_Slope** | Pente du segment ST (Up, Flat, Down) | Catégorielle |
| **HeartDisease** | **CIBLE** : 0 = Sain, 1 = Malade | Binaire |

---

## 🛠️ Méthodologie

### 1. Analyse Exploratoire (EDA)
- Analyse de la distribution des variables
- Identification des corrélations avec la cible
- Détection des valeurs manquantes et outliers
- Visualisations des relations clés

### 2. Prétraitement des Données

| Étape | Méthode | Description |
|-------|---------|-------------|
| **Valeurs manquantes** | Imputation par la médiane | 172 valeurs à 0 dans Cholesterol |
| **Outliers** | Winsorization (capping) | Bornes : Q1 - 1,5×IQR / Q3 + 1,5×IQR |
| **Encodage** | OneHotEncoder (drop='first') | Variables catégorielles → 9 features |
| **Standardisation** | StandardScaler | Moyenne = 0, Écart-type = 1 |
| **Split** | Train/Test (80/20) | 734 patients train / 184 patients test |

### 3. Modélisation

#### Modèles Testés

| Modèle | Type | Description |
|--------|------|-------------|
| **Logistic Regression** | Linéaire | Simple, rapide, interprétable |
| **Random Forest** | Ensemble (Bagging) | 100 arbres de décision |
| **Gradient Boosting** | Ensemble (Boosting) | Arbres séquentiels |
| **XGBoost** | Ensemble (Boosting) | Version optimisée |
| **SVM** | Marge | Séparateur à vaste marge |

#### Optimisation
- **Méthode** : GridSearchCV
- **Validation** : Cross-Validation 5-fold
- **Métrique** : F1-Score

---

## 📈 Résultats

### Tableau Comparatif des Modèles

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **89.13%** | **88.68%** | **92.16%** | **90.38%** | **93.33%** |
| Gradient Boosting | 88.59% | 89.32% | 90.20% | 89.76% | 91.89% |
| Random Forest | 88.04% | 87.74% | 91.18% | 89.42% | 92.32% |
| SVM | 87.50% | 86.24% | 92.16% | 89.10% | 94.23% |
| XGBoost | 84.78% | 87.00% | 85.29% | 86.14% | 91.49% |

### 🏆 Meilleur Modèle : Logistic Regression

**Performances Finales :**

```python
Accuracy  : 89.13%
Precision : 88.68%
Recall    : 92.16%  ← Le plus important en médecine !
F1-Score  : 90.38%
ROC-AUC   : 93.33%
