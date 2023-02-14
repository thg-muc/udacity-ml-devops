"""Module for performing the customer churn EDA and baseline model workflow.

This is part of the Udacity ML Devops Nanodegree (Project 1).

The main function triggers the default workflow. Further customization is
possible by modifying the constants.py file.
"""

# * Author  : Thomas Glanzer
# * Created : February 2023

# %% Libraries and Global Variables

import logging
import os

import pandas as pd
from generic_eda_library import GenericEdaLibrary

import constants

# %% Definitions


class ChurnLibrary(GenericEdaLibrary):
    """Add extra functionality for the Customer Churn Use Case."""

    def add_customer_churn(self, data) -> pd.DataFrame:
        """Make use-case specific adjustments to the data.

        Add a new column 'Churn' to the data, which is 1 if the customer
        has churned and 0 otherwise. The column 'Attrition_Flag' is used
        to determine the churn label.

        Parameters
        ----------
        data : pandas dataframe
            The dataframe to add the churn label to.

        Returns
        -------
        data : pandas dataframe
            The dataframe with new column 'Churn', without Attrition_Flag.
        """
        data['Churn'] = 0
        # ... fill it with 1 if "Attrited Customer"
        data.loc[data['Attrition_Flag'] ==
                 'Attrited Customer', 'Churn'] = 1
        # ... drop the old column
        data = data.drop('Attrition_Flag', axis=1)
        # ... define target
        return data

    def run_workflow(self):
        """Run the default workflow for this use-case."""
        # Initialize logging
        # ... check if path exists, if not create it
        os.makedirs(os.path.dirname(constants.DEFAULT_LOG_FILE), exist_ok=True)
        # ... set time, logger name, level and message
        logging.basicConfig(
            filename=constants.DEFAULT_LOG_FILE,
            level=constants.DEFAULT_LOG_LEVEL, filemode='w',
            format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        logging.info('Initialized customer churn workflow...')

        # Load data
        data = self.import_data(constants.DEFAULT_INPUT_FILE)
        # Make use case specific changes to the data
        data = self.add_customer_churn(data)

        # Perform EDA (statistics)
        # ... show DF head
        logging.info('EDA - Data Head...\n %s', data.head())
        # ... generate describe statistics with pandas
        logging.info('EDA - Describe...\n %s', data.describe())
        # ... generate isnull statistics with pandas
        logging.info('EDA - isnull sum...\n %s', data.isnull().sum())
        # EDA Plots: A combination of univariate and bivariate analysis
        # ... as well as a Correlation Matrix and stats written to logfile
        self.perform_eda(data, target=constants.TARGET)

        # Encode columns and get train test split from feature engineering
        train_test_data = self.perform_feature_engineering(
            data, target=constants.TARGET, feature_cols=constants.FEATURE_COLS,
            **constants.TRAIN_TEST_SPLIT_PARAMS)

        # Train and store models, generate plots and feature importances
        self.train_models(train_test_data, grid_dicts=constants.GRID_DICTS)

        # Give feedback that processing is done
        logging.info('All tasks finished successfully!')


if __name__ == '__main__':
    ChurnLibrary().run_workflow()
