# Predicting customer churn classification project

## Sources
**Python version:** 3.11.9<br/>
**Imported packages:** pandas, numpy, matplotlib, seaborn, sklearn, xgboost, statsmodels<br/>
**Dataset:** https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

## Data cleaning
After loading the data I checked duplicated observations and the missing (or incorrect) values for each feature.<br/>
The dataset was ready for EDA without any modifications.

## Exploratory Data Analysis
I prepared barplots for categorical features and histograms for numerical features to show insights about the distribution of Churn.

![Alt text](https://github.com/horvathadam07/customer-churn/blob/main/img/geography.PNG "Geography")

![Alt text](https://github.com/horvathadam07/customer-churn/blob/main/img/age.PNG "Age")

## Feature engineering
* New features created from existing ones like *HasJustCreditCard* from *NumOfProducts* and *HasCrCard*
* Dummy feature transformation for the first logistic regression model
* Weight of evidence encoding for the second logistic regression model


## Model Building
At first I splitted the data into train and test samples with a 20% test part.<br/>
I fitted six different models (with two logistic regressions) and evaluated them by weighted F1-score because the dataset is highly imbalanced.

The same cross validation (k=10) was used for all models to find the optimal probability threshold.<br/>
The goal was to maximize the weighted F1-score value in each iteration then the threshold was chosen as the average of the 10 train samples.

The tree based models were also optimized by GridSearchCV to find the best hyperparameters.

## Model Performances
The XGBoost algorithm performed significantly better than the others in the test sample, the weighted F1-scores can be seen below:

  * **Logistic regression (version dummy):** 0.7917<br/>
  * **Logistic regression (version woe):** 0.8145<br/>
  * **Decision tree:** 0.8088<br/>
  * **Random forest:** 0.8201<br/>
  * **Gradient boosting tree:** 0.8299<br/>
  * **XGBoost:** 0.8585

  Using weight of evidence encoding instead of dummy features is a better approach here for a logistic regression model.
