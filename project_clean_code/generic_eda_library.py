"""Generic Module for performing basic EDA and baseline model tasks.

This is part of the Udacity ML Devops Nanodegree (Project 1),
however it is written in a generic way to be used for other projects as well.
"""

# * Author  : Thomas Glanzer
# * Created : February 2023

# %% Libraries and Global Variables

import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay,
                             classification_report)
from sklearn.model_selection import GridSearchCV, train_test_split

# %% Definitions


class GenericEdaLibrary:
    """Generic library class for basic eda and baseline tasks.

    Attributes
    ----------
    image_path : string, default: 'images'
        The path to save the images to.

    model_path : string, default: 'models'
        The path to save the models to.

    log_path : string, default: 'logs'
        The path to save the logging output to.

    results_dict : dict, default: None
        Dictionary to store the results of the eda and baseline model.
        This is only available after model training.

    plot_style : string, default: 'darkgrid'
        The style to use for the plots (default: 'darkgrid').
        Reference: https://seaborn.pydata.org/tutorial/aesthetics.html
    """

    def __init__(self):
        # Default Attributes
        self.results_dict = None
        self.image_path = 'images'
        self.model_path = 'models'
        self.log_path = 'logs'
        self.plot_style = 'darkgrid'

    def import_data(self, input_file: str | None = None,
                    index_col: int = 0) -> pd.DataFrame:
        """Import data from a csv file and returns a pandas dataframe.

        The input_path is defined in the class initialization, but can be
        overwritten by providing a new path.

        Parameters
        ----------
        input_file : string
            The full path and file name to the input data file (csv)

        index_col : int or sequence or False, default 0
            Whether an index_col is present in the data which should be used.

        Returns
        -------
        data : pandas dataframe
            The input csv file as a pandas dataframe.
        """
        try:
            data = pd.read_csv(input_file, index_col=index_col)
            logging.info('File loaded with shape %s', data.shape)
        except FileNotFoundError:
            logging.error('File not found, please check the path.')
            raise

        return data

    def _get_cols_per_type(self, data: pd.DataFrame) -> tuple:
        """Return a dictionary with the column names per data type.

        Parameters
        ----------
        data : pandas dataframe
            The dataframe to determine the column types for.

        Returns
        -------
        cat_columns : list
            List of columns with categorical data (object, category, bool).

        quant_columns : list
            List of columns with quantitative data (int, float)

        other_columns : list
            List of columns with other data types
        """
        # Determine categorical and quantitative columns
        cat_columns = data.select_dtypes(
            include=['object', 'category', 'bool']).columns.tolist()
        quant_columns = data.select_dtypes(
            include=['int', 'float']).columns.tolist()
        # other columns are columns not in the above lists
        other_columns = [col for col in data.columns
                         if col not in cat_columns + quant_columns]

        return cat_columns, quant_columns, other_columns

    def perform_eda(self, data: pd.DataFrame, target: str,
                    nr_bins: int = 10) -> None:
        """Perform eda on a pandas dataframe and save results to output folder.

        Parameters
        ----------
        data : pandas dataframe
            The input dataframe to perform the eda on.

        target : string
            The name of the target column in the dataframe.

        nr_bins : int, optional, default 10
            The (max) number of bins to use for the histograms.

        Returns
        -------
        None
        """
        logging.info('Starting EDA...')
        # Show DF head
        logging.info('EDA - Data Head...\n %s', data.head())
        # Generate describe statistics with pandas
        logging.info('EDA - Describe...\n %s', data.describe())
        # Generate isnull statistics with pandas
        logging.info('EDA - isnull sum...\n %s', data.isnull().sum())

        # Determine column types
        cat_columns, quant_columns, _ = self._get_cols_per_type(data)

        # Set seaborn plot style
        with sns.axes_style(self.plot_style):
            # Generate hist plots figures for each column
            logging.info('EDA - Categorical Histograms...')
            for col in (cat_columns + quant_columns):
                # Determine number of bins and set defaults
                bins = min(nr_bins, data[col].nunique())
                x_data = data[col]
                target_mean = data[target].mean()
                kde = False
                shrink = .8
                # Set nr of bins for continuous columns
                if col in quant_columns:
                    shrink = 1
                    if bins == nr_bins:
                        x_data = pd.cut(
                            data[col], bins, precision=1, duplicates='drop')
                        kde = True
                # Create a matplotlib interface to generate figure and subplots
                _, (ax1, ax2) = plt.subplots(
                    nrows=1, ncols=2, figsize=(20, 10))
                # Default histplot
                sns.histplot(
                    x=col, data=data, ax=ax1, kde=kde,
                    bins=bins, hue=target, stat='percent',
                    multiple='stack', shrink=shrink).set(
                    title=f'Distribution of {col} (Count) '
                    f'- Population: {data.shape[0]}')
                # Barplot with target probability and confidence interval
                sns.barplot(x=x_data, data=data, y=target, ax=ax2,
                            capsize=.4, seed=42).set(
                    title=f'{target} Probability of {col} '
                    f'with Confidence Interval',
                    ylabel=f'{target} Probability')
                # ... add horizontal line to show the target probability
                ax2.axhline(
                    target_mean, color='red', linestyle='--',
                    label=f'{target} mean probability '
                    f'({100*target_mean:.1f}%)')
                ax2.legend(loc='upper right')
                # ... rotate x-ticks by 90
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
                # Save the figure and close it
                self._save_plot(
                    figure=plt, path=os.path.join(self.image_path, 'eda'),
                    plot_name='eda_' + col + '.png')
                plt.close('all')

        # Generate correlation matrix and plot
        logging.info('EDA - Correlation Matrix...')
        corr = data[quant_columns].corr()
        plt.figure(figsize=(20, 10))
        sns.heatmap(data=round(corr, 2), annot=True, cmap='Dark2_r',
                    linewidths=2).set_title("Correlation Matrix")
        self._save_plot(
            figure=plt, path=os.path.join(self.image_path, 'eda'),
            plot_name='eda__Correlation_Matrix.png')
        plt.close('all')

    def _save_plot(self, figure: plt, path: str,
                   plot_name: str, **kwargs) -> None:
        """Save current plot to file and create folders if required.

        Parameters
        ----------
        figure : matplotlib.pyplot
            The figure to save

        path : string
            The path where the plot should be saved

        plot_name : string
            Filename to use for the plot to save (in class output_path)

        **kwargs : optional
            Additional keyword arguments to pass to plt.savefig()

        Returns
        -------
        None
        """
        # Make sure the given output path exists
        os.makedirs(path, exist_ok=True)
        # Use default args, but update with provided params
        plot_kwargs = {'bbox_inches': 'tight'}
        plot_kwargs.update(**kwargs)
        # Save the figure
        figure.savefig(os.path.join(path, plot_name), **plot_kwargs)

    def perform_feature_engineering(
            self, data: pd.DataFrame, target: str,
            feature_cols: list | None = None, **kwargs) -> tuple:
        """Encode categorical features and perform a train test split.

        Parameters
        ----------
        data : pandas dataframe
            The input dataframe for feature engineering.

        target : string
            The target label used for prediction training and evaluation

        feature_cols : list, optional, default None
            List of feature columns to keep, other columns will be dropped.

        **kwargs : dict
            Additional keyword arguments for the train test split

        Returns
        -------
        train_test_data : tuple
            Tuple containing the training and test data and target values.
            Expected order: (data_train, data_test, y_train, y_test)
        """
        # Encode categorical columns into new columns with target proportions
        data = self.encoder_helper(data=data, target=target)

        # .. and perform the train test split
        train_test_data = train_test_split(
            data[feature_cols], data[target], **kwargs)

        return train_test_data

    def encoder_helper(self, data: pd.DataFrame, target: str,
                       keep_original=False) -> pd.DataFrame:
        """Encode categorical columns into new columns with target proportions.

        This is helper function to turn each categorical column into a new
        column with target propotion for each category.

        Parameters
        ----------
        data : pandas dataframe
            The input dataframe for encoding

        target : string
            The target label used for encoding the categorical features

        keep_original : bool, optional, default False
            Whether to keep the original categorical columns or not

        Returns
        -------
        data : pandas dataframe
            dataframe with additional encoded columns
        """
        # Determine categorical and quantitative columns
        cat_cols, _, _ = self._get_cols_per_type(data)
        # Iterate over the categorical columns
        for cat in cat_cols:
            # Get the target proportion for each category
            cat_means = data.groupby(cat)[target].mean(numeric_only=True)
            # Iterate over the means, add value per group and add a new column
            for cat_mean_group in cat_means.index:
                data.loc[data[cat] == cat_mean_group,
                         cat + f'_{target}'] = cat_means[cat_mean_group]
        # Drop the original categorical columns if not requested
        if not keep_original:
            data = data.drop(cat_cols, axis=1, errors='ignore')

        return data

    def train_models(self, train_test_data: tuple, grid_dicts: dict,
                     **kwargs) -> None:
        """Use Training and Test data to train and evaluate models.

        This will train two separate models (logistic regression and
        random forest) and persist the results (images, scores, model files) to
        make it easier to compare the models.

        Parameters
        ----------
        train_test_data : tuple
            Tuple containing the training and test data and target values.
            Expected order: (data_train, data_test, y_train, y_test)

        grid_dicts : dict
            Dictionary containing grid search dicts for the models.

        **kwargs : dict
            Additional keyword arguments for GridSearchCV.

        Returns
        -------
        None
        """
        # Unpack the data
        data_train, data_test, y_train, y_test = train_test_data
        # Define the baseline models
        lrc = LogisticRegression()
        rfc = RandomForestClassifier()

        # Define the parameter grid for the random forest
        lrc_param_grid = grid_dicts['lrc']
        rfc_param_grid = grid_dicts['rfc']

        # Train the models
        cv_lrc = GridSearchCV(
            estimator=lrc, param_grid=lrc_param_grid, **kwargs)
        cv_lrc.fit(data_train, y_train)

        cv_rfc = GridSearchCV(
            estimator=rfc, param_grid=rfc_param_grid)
        cv_rfc.fit(data_train, y_train)

        # Create prediction result dictionary
        self.results_dict = results_dict = {
            'LogisticRegression': {
                'model': cv_lrc.best_estimator_,
                'y_train': y_train,
                'y_test': y_test,
                'y_train_preds': cv_lrc.best_estimator_.predict(data_train),
                'y_test_preds': cv_lrc.best_estimator_.predict(data_test),
                'y_train_proba': cv_lrc.best_estimator_.predict_proba(
                    data_train)[:, 1],
                'y_test_proba': cv_lrc.best_estimator_.predict_proba(
                    data_test)[:, 1]
            },
            'RandomForest': {
                'model': cv_rfc.best_estimator_,
                'y_train': y_train,
                'y_test': y_test,
                'y_train_preds': cv_rfc.best_estimator_.predict(data_train),
                'y_test_preds': cv_rfc.best_estimator_.predict(data_test),
                'y_train_proba': cv_rfc.best_estimator_.predict_proba(
                    data_train)[:, 1],
                'y_test_proba': cv_rfc.best_estimator_.predict_proba(
                    data_test)[:, 1]
            }
        }

        # Store models to disk
        self._store_models(results_dict)

        # Create Performance Metric Plots into Results folder
        self._plot_performance_curves(results_dict)

        # Create Classification Report Plots into Results folder
        self._classification_report_image(results_dict)

        # Generate feature importances plots
        self._plot_feature_importances(results_dict)

    def _store_models(self, results_dict: dict) -> None:
        """Store the trained models in the output path.

        Parameters
        ----------
        results_dict : dict
            Dictionary with the trained models

        Returns
        -------
        None
        """
        # Store the models to disk
        for model_name, model_results in results_dict.items():
            model = model_results['model']
            model_path = os.path.join(self.model_path, f'{model_name}.pkl')
            # ... Create the model directory if it does not exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)

    def _plot_performance_curves(self, results_dict: dict) -> None:
        """Create performance curve plots and store in a path.

        Parameters
        ----------
        results_dict : dict
            Dictionary containing the training results (model, score, etc.)

        Returns
        -------
        None
        """
        # Iterate over all different performance plots
        for plot_name, Display in [
                ('ROC_AUC', RocCurveDisplay),
                ('Precision_Recall', PrecisionRecallDisplay)]:
            #ax1 = plt.subplot()
            plt.figure(figsize=(12, 8))
            ax1 = plt.gca()
            # Iterate over all different models (results_dict)
            for model_name, model_results in results_dict.items():
                # Get the model and predictions
                model = model_results['model']
                model_name = type(model).__name__
                y_test = model_results['y_test']
                y_test_proba = model_results['y_test_proba']

                # Create the plot
                Display.from_predictions(
                    y_test, y_test_proba, ax=ax1, name=model_name)
            # Add chance line for roc auc curve
            if plot_name == 'ROC_AUC':
                ax1.plot([0, 1], [0, 1], linestyle='--',
                         label='Chance', color='g')
            plt.legend()
            plt.title(f'{plot_name} Curve for selected Models')
            plt.grid(True)
            self._save_plot(
                figure=plt, path=os.path.join(self.image_path, 'results'),
                plot_name=f'{plot_name}_Curve.png')
            plt.close('all')

    def _classification_report_image(self, results_dict: dict) -> None:
        """Use test and train data to create classification reports.

        Classification reports for both training and testing results will be
        created, the report will be stored to the output path.

        Parameters
        ----------
        results_dict : dict
            Dictionary containing the training results (model, score, etc.)

        Returns
        -------
        None
        """
        # Iterate over all different models (results_dict)
        for model_name, model_results in results_dict.items():
            y_test = model_results['y_test']
            y_train = model_results['y_train']
            y_test_preds = model_results['y_test_preds']
            y_train_preds = model_results['y_train_preds']

            # Create the plot from classification report
            plt.rc('figure', figsize=(5, 5))
            plt.text(0.01, 1.25, str(f'{model_name} Train'), {
                'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.05, str(classification_report(y_test, y_test_preds)),
                {'fontsize': 10},
                fontproperties='monospace')
            plt.text(0.01, 0.6, str(f'{model_name} Test'), {
                'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.7, str(classification_report(y_train, y_train_preds)),
                {'fontsize': 10},
                fontproperties='monospace')
            plt.axis('off')

            self._save_plot(
                figure=plt, path=os.path.join(self.image_path, 'results'),
                plot_name=f'{model_name}_ClassificationReport.png')
            plt.close('all')

    def _plot_feature_importances(self, results_dict: dict) -> None:
        """Create feature importances plots from model and store in a path.

        Parameters
        ----------
        results_dict : dict
            Dictionary containing the training results (model, score, etc.)

        Returns
        -------
        None
        """
        # Iterate over models for Feature Importances plot
        for model_name, model_results in results_dict.items():

            model = model_results['model']
            # ... Try if model offers feature importances
            try:
                importances = model.feature_importances_
            except AttributeError:
                logging.info(
                    '%s does not offer feature importances.', model_name)
                continue

            # ... Sort feature importances in descending order
            indices = np.argsort(importances)[::-1]
            # Get features and rearrange them
            features = model.feature_names_in_
            names = [features[i] for i in indices]

            # ...Create seaborn plot
            with sns.axes_style(self.plot_style):
                plt.figure(figsize=(20, 5))
                sns.barplot(x=names, y=importances[indices]).set_title(
                    f"Feature Importances for {model_name} Model")
                plt.ylabel('Feature Importance')
                _ = plt.xticks(rotation=90)
            # ... Save plot to path
            self._save_plot(
                figure=plt, path=os.path.join(self.image_path, 'results'),
                plot_name=f'{model_name}_FeatureImportances.png')
            plt.close('all')


# %%

# TODO check logging consistency with other modules
