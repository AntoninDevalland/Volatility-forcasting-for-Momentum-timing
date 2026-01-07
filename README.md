# Prévision de la volatilité et market timing du facteur Momentum

Prévision de la volatilité du facteur Momentum à l’aide de modèles économétriques et de machine learning, et évaluation de leur impact sur la performance et le contrôle du risque à travers des stratégies de volatility targeting.

## Objectif

Ce projet étudie la prévision de la volatilité du facteur Momentum à partir de modèles économétriques classiques (AR, HAR, GARCH) et de méthodes de machine learning (Elastic Net, LightGBM).  
L’objectif est d’évaluer à la fois la qualité statistique des prévisions et leur utilité économique à travers des stratégies de gestion dynamique du risque.

## Contenu du dépôt

- `momentum_volatility.ipynb` : notebook principal contenant
  - la construction des données et des variables explicatives,
  - l’estimation des modèles de prévision de la volatilité,
  - l’évaluation des performances hors échantillon,
  - l’application à des stratégies de volatility targeting.

## Méthodologie

- Prévision de la variance réalisée hebdomadaire du facteur Momentum  
- Comparaison de modèles économétriques et de modèles de machine learning  
- Évaluation hors échantillon des performances prédictives  
- Application des prévisions à des stratégies d’allocation dynamique du risque

## Données

Les données utilisées incluent :
- le facteur Momentum et les facteurs Fama–French (Kenneth French Data Library),
- des variables macro-financières et de marché (taux d’intérêt, spreads de crédit, VIX, taux de change).

Toutes les variables sont observables en temps réel afin d’éviter toute fuite d’information.

## Utilisation

Le projet est entièrement contenu dans un notebook Jupyter.  
Il peut être exécuté séquentiellement cellule par cellule.

Principales dépendances :
- `pandas`
- `numpy`
- `scikit-learn`
- `statsmodels`
- `lightgbm`

## Résultats principaux

- Les stratégies de volatility targeting permettent de presque doubler le ratio de Sharpe du facteur Momentum.
- Les modèles de machine learning apportent des améliorations modestes mais robustes par rapport aux benchmarks économétriques.

## Remarque

Ce projet est réalisé dans un cadre académique et ne constitue pas une recommandation d’investissement.
