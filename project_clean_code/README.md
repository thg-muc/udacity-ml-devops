# Predict Customer Churn

## Project Description

This is a solution for the project **Predict Customer Churn** of the Udacity ML DevOps Engineer Nanodegree. An exemplary Jupyter notebook was provided, performing both EDA and Baseline Model training tasks. A valid solution should refactor this notebook, break it up into multiple scripts and make improvements to the code base wherever possible. Also, logging and testing functionality is to be added.

## Solution Description

This solution is loosely based on the provided notebook, however multiple remarkable changes were made to the overall code-base and structure:

- An **object oriented approach** was followed, refactoring all code (considering coding best practices) and moving all generic and reusable code into a class called `GenericEdaLibrary`.
- This generic class is inherited by the `ChurnLibrary` class, which contains all the code for the EDA and Baseline Model training tasks, but also adds some additional use-case specific functionality.
- The EDA part of the original Notebook was extended and improved, generating **many additional univariate and bivariate plots for each feature**.
- The Baseline Model training part of the original Notebook was extended and improved, generating additional plots for the model performance (**Precision-Recall curve in addition to the ROC-AUC curve**). Via the configuration file it is possible to define the search grid for both models for the hyperparameter tuning.
- All EDA and Result plots are persisted to disk. The design and layout of most plots was improved by tuning the plot parameters and using `seaborn` library.
- Logging functionality was added to the code, using the `logging` library.
- TestCases were added in a separate script using the `pytest` library. The test cases directly invoke all public methods from the classes and check the results against expected values. Indirectly, all private methods are also tested, as they are invoked by the public methods.
- In general the code was refactored and improved, **considering coding best practices and making the code more readable and maintainable**. A pylint Score of 10.0 was achieved in each python script.

## Files and data description

The following files are present in the root directory:

- `churn_library.py`: This file extends the `GenericEdaLibrary` class with use-case specific functionality and contains the default workflow for the EDA and Baseline Model training tasks.
- `churn_script_logging_and_tests.py`: Contains the code for logging and testing the churn_library.
- `constants.py`: Configuration file for the churn_library.
- `generic_eda_library.py`: This file contains the functions that are used in the notebook.
- `pyproject.toml`: Configuration file for the project. This is were pytest and its logging are configured.
- `README.md`: This file.
- `requirements.txt`: Contains the requirements for the project. Install them with *pip install -r requirements.txt*.

## Libraries

The following libraries are used in this project in conjunction with Python 3.10.8:

    - autopep8==2.0.0
    - joblib==1.2.0
    - matplotlib==3.6.3
    - numpy==1.23.5
    - pandas==1.5.3
    - pylint==2.15.10
    - pytest==7.2.1
    - scikit_learn==1.2.1
    - seaborn==0.12.2

Consider using a virtual environment and *requirements.txt* to install the libraries.

## Running Files

a) To execute the main customer_churn workflow, consisting of EDA + Baseline Model training, run the following command inside the project directory:

    python churn_library.py

→ The EDA and Baseline Model training results are persisted to disk in the `images` and `models` folder. A result log - containing additional textual information -  will be created in `logs\churn_library.log`.

b) To execute the logging and testing workflow, run the following command:

    pytest churn_script_logging_and_tests.py

→ This executes all pytest-tests for the churn_library by invoking all public methods from the classes. The test results are logged to the console and to the file `logs\pytest.log` - the pytest logging configuration is set in the `pyproject.toml` file. This logfile only contains the test results, so it is not redundant to the primary logfile.

By running the functions of the churn_library, the EDA and Baseline Model training results are also persisted to disk in the `images` and `models` folder. Please note that on order to evaluate that the expected results were created, these folders and their contents are deleted before the tests are executed.
