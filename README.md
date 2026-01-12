# Prévision de la volatilité et market timing du facteur Momentum

Prévision de la volatilité du facteur Momentum à l’aide de modèles économétriques et de machine learning, et évaluation de leur impact sur la performance et le contrôle du risque à travers des stratégies de volatility targeting.

## Objectif

Ce projet étudie la prévision de la volatilité du facteur Momentum à partir de modèles économétriques classiques (AR, HAR, GARCH) et de méthodes de machine learning (Elastic Net, LightGBM). 

L’objectif est d’évaluer à la fois la qualité statistique des prévisions et leur impact sur la performance et le contrôle du risque à travers des stratégies de gestion dynamique du risque.

## Contenu du dépôt

- `momentum_volatility.ipynb` : notebook principal.
- `Volatilite_Momentum_Market_Timing` : rapport détaillé présentant la méthodologie, les résultats empiriques et leur interprétation économique.
- autres fichiers et scripts : fonctions utilitaires et modules auxiliaires utilisés par le notebook principal.

## Méthodologie

- Prévision de la variance réalisée hebdomadaire du facteur Momentum  
- Comparaison de modèles économétriques et de modèles de machine learning  
- Évaluation hors échantillon des performances prédictives  
- Application des prévisions à des stratégies d’allocation dynamique du risque de type volatility targeting.
- Réallocation dynamique entre le facteur Momentum et des actifs de repli tels que les Treasury Bills ou des facteurs actions défensifs (CMA, RMW, etc.).

## Données

Les données sont journalières (1990-2025) et incluent :
- le facteur Momentum et les facteurs Fama–French (Kenneth French Data Library),
- des variables macro-financières et de marché (taux d’intérêt, spreads de crédit, VIX, taux de change).

Toutes les variables sont observables en temps réel afin d’éviter toute fuite d’information.

## Résultats principaux

- Les stratégies de volatility targeting permettent de presque doubler le ratio de Sharpe du facteur Momentum.
- Les modèles de machine learning apportent des améliorations de prévision par rapport aux benchmarks économétriques.

