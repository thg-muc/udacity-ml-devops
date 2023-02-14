"""Tests for churn_library.py EDA module.

This is part of the Udacity ML Devops Nanodegree (Project 1).

The test cases directly invoke all public methods from the classes
and check the results against expected values. Indirectly, all private methods
are also tested, as they are invoked by the public methods.
"""

# * Author  : Thomas Glanzer
# * Created : February 2023

# %% Libraries and Global Variables

import glob
import logging
import os
import shutil

import pandas as pd
import pytest
from churn_library import ChurnLibrary

import constants

# %% Classes and Functions

# NOTE: pytest logging is configured in the pyproject.toml file


@pytest.fixture(scope="session", autouse=True)
def setup_tests() -> None:
    """Prepare tests by deleting models/images folders and contents."""
    logging.info(
        'Setting up tests by deleting models/logs/images folders...')
    cls = ChurnLibrary()
    for folder in [cls.model_path, cls.image_path]:
        shutil.rmtree(folder, ignore_errors=True)
    logging.info('Starting tests for churn_library.py EDA module...')


@pytest.fixture(name='cls')
def fixture_init_class() -> ChurnLibrary:
    """Initialize the ChurnLibrary class."""
    try:
        cls = ChurnLibrary()
    except Exception as err:
        raise err
    return cls


@pytest.fixture(name='raw_data')
def fixture_import_data(cls: ChurnLibrary) -> pd.DataFrame:
    """Import raw test data from file."""
    try:
        raw_data = cls.import_data(constants.DEFAULT_INPUT_FILE)
    except Exception as err:
        raise err
    return raw_data


@pytest.fixture(name='data')
def fixture_preprocess_data(cls: ChurnLibrary,
                            raw_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the raw data."""
    try:
        data = cls.add_customer_churn(raw_data)
    except Exception as err:
        raise err
    return data


@pytest.fixture(name='train_test_data')
def fixture_feature_engineering(
        cls: ChurnLibrary, data: pd.DataFrame) -> tuple:
    """Perform feature engineering and return a train/test data split."""
    try:
        train_test_data = cls.perform_feature_engineering(
            data, target=constants.TARGET, feature_cols=constants.FEATURE_COLS,
            **constants.TRAIN_TEST_SPLIT_PARAMS)
    except Exception as err:
        raise err
    return train_test_data


def test_init_class(cls: ChurnLibrary) -> None:
    """Test the initialization of the ChurnLibrary class."""
    try:
        assert cls is not None
        assert cls.__class__.__name__ == 'ChurnLibrary'
        logging.info("Testing ChurnLibrary: SUCCESS - Class Initialized")
    except AssertionError as err:
        logging.error("Testing ChurnLibrary: FAILED - not Initialized")
        raise err


def test_import_data(raw_data: pd.DataFrame):
    """Test the import_data function."""
    try:
        assert raw_data is not None
        assert raw_data.shape[0] > 0  # Check that file is not empty
        assert raw_data.shape[1] > 1  # Check that separator is correct
        # Check that header is correct
        assert raw_data.columns[0] == 'CLIENTNUM'
        logging.info("Testing import_data: SUCCESS - File Loaded")
    except AssertionError as err:
        logging.error(
            """Testing import_data: FAILED - File does not have the
                        expected rows/cols/header.""")
        raise err


def test_eda(cls: ChurnLibrary, data: pd.DataFrame) -> None:
    """Test the perform_eda function."""
    try:
        eda_path = f'{cls.image_path}{os.sep}eda{os.sep}'
        # Identify features columns to get nr of expected plots
        nr_features = len(data.select_dtypes(
            include=['object', 'category', 'bool', 'int', 'float'])
            .columns.tolist())
        cls.perform_eda(data, constants.TARGET)
        # Check Correlation Matrix image was created
        assert len(
            glob.glob(eda_path + '*Correlation_Matrix.png')) >= 1
        # Check one png was created per feature (in addition to CorrMatrix)
        assert len(
            glob.glob(eda_path + '*.png')) > nr_features

        logging.info("Testing perform_eda: SUCCESS - EDA performed")
    except Exception as err:
        logging.error("Testing perform_eda: FAILED - EDA not performed")
        raise err


def test_encoder_helper(cls: ChurnLibrary, data: pd.DataFrame) -> None:
    """Test the encoder_helper function."""
    try:
        data = cls.encoder_helper(data, target=constants.TARGET)
        # Assert that there are no categorical columns left after encoding
        assert len(data.select_dtypes(
            include=[
                'object', 'category', 'bool']).columns) == 0
        logging.info(
            "Testing encoder_helper: SUCCESS - Encoding performed")
    except Exception as err:
        logging.error(
            "Testing encoder_helper: FAILED - Encoding not performed")
        raise err


def test_perform_feature_engineering(train_test_data: tuple) -> None:
    """Test the perform_feature_engineering function."""
    try:
        assert train_test_data is not None
        # Check that the train_test_data tuple has 4 elements
        assert len(train_test_data) == 4
        # Assert first two elements are pd DataFrames (X_train, X_test)
        assert all(isinstance(x, pd.DataFrame)
                   for x in train_test_data[:2])
        # Assert last two elements are pandas Series (y_train, y_test)
        assert all(isinstance(x, pd.Series) for x in train_test_data[2:])

        logging.info("""Testing perform_feature_engineering: SUCCESS -
                        Feature Engineering performed""")
    except Exception as err:
        logging.error("""Testing perform_feature_engineering: FAILED -
                        Feature Engineering not performed""")
        raise err


def test_train_models(cls: ChurnLibrary, train_test_data: tuple) -> None:
    """Test the train_models function."""
    try:
        cls.train_models(
            train_test_data, grid_dicts=constants.GRID_DICTS)

        # Assert that the results_dict was initialized
        assert cls.results_dict is not None
        # Assert there are two keys "RandomForest" and "LogisticRegression"
        assert all(key in cls.results_dict.keys()
                   for key in ['RandomForest', 'LogisticRegression'])
        # Assert that at least two models were stored in the model folder
        assert len(glob.glob(cls.model_path + f'{os.sep}*.pkl')) >= 2

        logging.info("""Testing train_models: SUCCESS -
                        Models trained and stored.""")
    except Exception as err:
        logging.error("""Testing train_models: FAILED -
                        Models not trained and stored.""")
        raise err


# %%
