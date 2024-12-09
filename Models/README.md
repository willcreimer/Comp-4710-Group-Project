# Wildfires classification

## Overview

This repository presents the work described in the paper "Wildfire causes prediction using classification techniques - A comparative study". The paper explores the use of machine learning techniques to classify wildfire causes based on various environmental and temporal features. By leveraging historical data. This work aims to contribute to wildfire prevention efforts and better resource management.

## Approach

1. Data Collection:  
   We used the of Historical wildfire data from 2006 to 2023 in Alberta available [here](https://open.canada.ca/data/en/dataset/a221e7a0-4f46-4be7-9c5a-e29de9a3447e). 2 datasets are included, `data-cleaned` contains entries with all the irrelevant features removed but still has missing value. `data-cleaned-removed-empty` contains entries where all such missing value entries are removed.
2. Preprocessing:  
   Missing values are handled either by adding "N/A" for categorical features or using the medial for numerical features.
3. Model Selection:
   The following are models selected for our works.
   - Logistic Regression
   - Support Vector Machine
   - Gradient Boost Tree
   - XGBoost
   - Decision Tree
   - Random Forest
   - Voting Classifier
   - Stacking Classifier

## Code

For .ipynb files, you can either run them on Jupyter Lab, google collab or the Jupyter VSCode extension.  
For .py file, simply run them with python ./\<model.py\>  
Note that the result for these models are stochastic in nature due to the random split of training and testing dataset.
Code for hyperparameter tuning using GridSearchCV is turned off by default since it takes too long.

## Requirement

```
pandas
numpy
sklearn
imblearn
```
