import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from census_income.exception.exception import CustomException
from census_income.utils.utils import save_object, evaluate_model



# Intitialize the Model Trainer Configuration
@dataclass
class ModelTrainerConfig:
    # Create pickle file path for storing model
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    # Create pickle file path for storing best model
    trained_model_grid_search_file_path = os.path.join('artifacts', 'model_grid_search.pkl')


# create a class for Model Trainer
class ModelTrainer:
    """
    This class is used to store the configuration of the model trainer
    """

    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def intitate_model_training(self, train_array, test_array):
        """
        This method is used to get the model training
        """

        try:
            # Create a train and test array for training a data
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )

            # Create Logistic Regression Model
            model = LogisticRegression()

            # Train LogisticRegression Model
            model.fit(X_train, y_train)

            # Get Performance Report 
            model_report:dict=evaluate_model(X_train,X_test,y_train,y_test,model)
            

            print(model_report)
            print('\n====================================================================================\n')
            # Perform Grid Search to find the best hyper parameters for Optimization

            # Create a dictionary of hyper parameters
            param_grid = {
                'penalty': ['l1', 'l2', 'elasticnet'],
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear', 'saga']
            }

            # Create a GridSearchCV Model
            grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='roc_auc')

            # Train GridSearchCV Model
            grid_search.fit(X_train, y_train)

            # To get best parameters from dictionary 
            best_model_parameters = grid_search.best_params_

            # Get Performance Report 
            model_report:dict=evaluate_model(X_train,X_test,y_train,y_test,grid_search)

            print(f'After applying GridSearchCV R2 Score : {model_report['r2 Score']}, ROC AUC Score : {model_report['roc_auc Score']}')
            print(f'Best Parameters for Logistic Regression are : {best_model_parameters}')

            print('\n====================================================================================\n')

            # Save the trained model
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=model
            )

            # Save the best trained model
            save_object(
                 file_path=self.model_trainer_config.trained_model_grid_search_file_path,
                 obj=grid_search
            )


        except Exception as e:
            raise CustomException(e, sys) from e