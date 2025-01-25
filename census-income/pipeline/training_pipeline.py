import os
import sys

from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer

import pandas as pd


if __name__=='__main__':
    # For Data Ingestion
    dat_ingestion = DataIngestion()
    train_data_path, test_data_path = dat_ingestion.initiate_data_ingestion()

    # For Data Transformation
    dat_transformation = DataTransformation()
    train_data, test_data, _ = dat_transformation(train_data_path, test_data_path)

    # For Model Training
    model_trainer = ModelTrainer()
    model_trainer.train_model(train_data, test_data)