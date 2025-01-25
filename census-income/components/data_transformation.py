import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from exception.exception import CustomException

from utils.utils import save_object




# Intitialize the Data Transformation Configuration
@dataclass
class DataTransformationconfig:
    # Create pickle file path for storing preprocessing data
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# create a class for Data Transformation
class DataTransformation:
    """
    This class is used to store the configuration of the data transformation
    """

    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_obj(self):
        """
        This method is used to get the data transformation object
        """

        try:
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Workclass', 'Occupation', 'Native Country']
            numerical_cols = ['Age', 'Final Weight', 'EducationNum', 'Capital Gain', 'capital loss', 'Hours per Week']

            # create a Pipeline

            # For numerical columns
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            # For categorical columns
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
            ])

            # Preprocessing
            preprocessor = ColumnTransformer(transformers=[
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ])

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This method is used to initiate the data transformation
        """

        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Get the data transformation object
            data_transformation_obj = self.get_data_transformation_obj()

            # Select Target Column
            target_column_name = 'Income'

            # For Input Features remove the target column from train and test data
            independent_feature_train_df = train_df.drop(target_column_name, axis=1)
            independent_feature_test_df = test_df.drop(target_column_name, axis=1)

            # For Target Variable remove the another column from train and test data
            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            # Apply the data transformation to the train and test data
            input_train_df = data_transformation_obj.fit_transform(independent_feature_train_df)
            input_test_df = data_transformation_obj.transform(independent_feature_test_df)

            # Final train and test data
            train_data = np.c_[input_train_df, np.array(target_feature_train_df)]
            test_data = np.c_[input_test_df, np.array(target_feature_test_df)]

            # Data store in pickle file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = data_transformation_obj
            )

            return(
                train_data,
                test_data,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            
        except Exception as e:
            raise CustomException(e, sys)