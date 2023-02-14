"""Define constants and default settings for churn_library.py Module.

This is part of the Udacity ML Devops Nanodegree (Project 1).
"""

# * Author  : Thomas Glanzer
# * Created : February 2023

# %% Libraries and Global Variables

import os

#######################################
# Basic Configuration for churn_library

DEFAULT_INPUT_FILE = os.path.join('data', 'bank_data.csv')
DEFAULT_LOG_FILE = os.path.join('logs', 'churn_library.log')
DEFAULT_LOG_LEVEL = 'INFO'

#######################################
# Baseline Model Options

TARGET = 'Churn'

# Columns used as Input for the Model
FEATURE_COLS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn']

# Define Train_Test_Split for training
TRAIN_TEST_SPLIT_PARAMS = dict(
    test_size=.3,
    random_state=42
)

# Define the RF Grid Search
RF_GRID_SEARCH_PARAMS = dict(
    n_estimators=[200, 500],
    criterion=['gini', 'entropy'],
    max_features=[None, 'sqrt'],    # sqrt and auto are the same
    max_depth=[5, 8, 13, 20, None],
    min_samples_leaf=[1, 2, 3],
    random_state=[42],
    n_jobs=[8],
)

# Define the LogisticRegression Grid Search
LR_GRID_SEARCH_PARAMS = [
    dict(
        solver=['liblinear'],
        penalty=['l1', 'l2'],
        C=[0.1, 1, 10],
        max_iter=[3000],
        random_state=[42],
    ),
    dict(
        solver=['lbfgs'],
        penalty=['l2'],
        C=[0.1, 1, 10],
        max_iter=[3000],
        random_state=[42],
        n_jobs=[8],
    ),
]

# Create a combined Grid Dict
GRID_DICTS = dict(
    rfc=RF_GRID_SEARCH_PARAMS,
    lrc=LR_GRID_SEARCH_PARAMS
)
