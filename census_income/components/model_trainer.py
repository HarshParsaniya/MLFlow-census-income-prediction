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
            # Create compatible parameter grid
            # param_grid = [
            #     {
            #         'penalty': ['l2'],  # Use l2 explicitly
            #         'solver': ['lbfgs'],
            #         'C': [0.001, 0.01, 0.1, 1, 10],
            #         'class_weight': [None, 'balanced']
            #     },
            #     {
            #         'penalty': ['l1', 'l2'],
            #         'solver': ['liblinear'],
            #         'C': [0.001, 0.01, 0.1, 1, 10],
            #         'class_weight': [None, 'balanced']
            #     }
            # ]

            # # Create a GridSearchCV Model
            # grid_search = GridSearchCV(
            #     estimator=LogisticRegression(max_iter=1000),
            #     param_grid=param_grid,
            #     scoring=['accuracy', 'f1', 'roc_auc'],
            #     refit='f1',
            #     cv=5,
            #     n_jobs=-1,
            #     verbose=2,
            #     error_score='raise'
            # )

            # Expanded parameter grid with more comprehensive options
            param_grid = [
                {
                    'penalty': ['l2'],
                    'solver': ['lbfgs'],
                    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    'class_weight': [None, 'balanced'],
                    'max_iter': [500, 1000, 1500]
                },
                {
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear'],
                    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    'class_weight': [None, 'balanced']
                }
            ]

            # Create a GridSearchCV Model
            grid_search = GridSearchCV(
                estimator=LogisticRegression(),
                param_grid=param_grid,
                scoring=['accuracy', 'f1_weighted', 'roc_auc'],
                refit='f1_weighted',  # Change refit metric
                cv=5,
                n_jobs=-1,
                verbose=2,
                error_score='raise'
            )

            # Train GridSearchCV Model
            grid_search.fit(X_train, y_train)

            # To get best parameters from dictionary 
            best_model_parameters = grid_search.best_params_

            # Get Performance Report 
            best_model_report:dict=evaluate_model(X_train,X_test,y_train,y_test,grid_search)

            print(f'After applying GridSearchCV Accuracy Score : {best_model_report['accuracy Score']}, ROC AUC Score : {best_model_report['roc_auc Score']}')
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