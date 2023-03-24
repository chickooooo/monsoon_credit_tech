# Monsoon CreditTech Assignment

<br>
<br>

## Project Structure

```
root
    - /notebooks                       jupyter notebooks (in order)
    - /pipeline_data:                  data generated when passed through each data pipeline (in order)
    - /scripts/pipelines:              python scripts for data pipelines (in order)
    - /scripts/utils.py                utility functions
    - /Test/final_predictions.csv      final predictions file (predictions on test data)
    - /Test/X_test.csv                 provided test data
    - /Training                        provided training data
    - /notes.txt                       general notes about the project
    - /Problem Statement.docx          problem statement
    - /README.md                       readme file
    - /sample_submission_file.csv      sample submission file

```

<br>
<br>

## Tasks performed in individual jupyter notebook:

- `EDA`:
    - data analysis

- `Feature Selection`:
    - removal of informational features
    - removal of features with high missing values
    - removal of features with high correlation

- `Data Preparation`:
    - reducing cardinality of categorical features
    - handling outliers in numerical features
    - missing values imputation
    - transformation of skewed nuemrical features

- `Feature Engineering`:
    - na

- `Data Preprocessing`:
    - encoding of boolean & categorical features
    - scaling of numerical features

- `Feature Selection 2`:
    - recursive feature elimination

- `Data Pipeline Tuning`:
    - tuning parameters of above data pipelines

- `Model Selection`:
    - hyperparameter tuning

- `Test Data Performance`:
    - verifying data pipelines
    - checking model performance on test data (holdout)

- `Making Prediction`:
    - training model on complete training data
    - making predictions on test data
    - creating submission file

<br>
<br>

## Project Stats

```
domain:                Machine Learning
sub-domain:            -
problem category:      Binary Classification
approach:              Supervised Machine Learning
model:                 LightGBM
evaluation metric:     ROC AUC
score:                 0.755
```
