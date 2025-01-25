import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from exception.exception import CustomException

# Intitialize the Data Ingetion Configuration
@dataclass
class DataIngestionconfig:
    # Create raw dataset path
    raw_data_path = os.path.join('artifacts', 'raw.csv')
    # Create train dataset path
    train_data_path = os.path.join('artifacts', 'train.csv')
    # Create test dataset path
    test_data_path = os.path.join('artifacts', 'test.csv')


# create a class for Data Ingestion
class DataIngestion:
    """
    This class is used to store the configuration of the data ingestion
    """

    def __init__(self) -> None:
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        """
        This method is used to create raw, train, and test dataset
        """
        
        try:
            # Read the Data from Original Dataset
            df = pd.read_csv(os.path.join('notebook/data/census-income', 'adult.csv'))

            # Create an artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            # Create raw dataset at artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # Drop an Unnecessary(Noisy) Data
            df.drop(['Education','Marital Status', 'Relationship', 'Race', 'Gender'],axis=1,inplace=True)

            # Clean a Wrong Data
            df['Workclass'] = df['Workclass'].str.strip()
            df['Workclass'] = df['Workclass'].replace('?', np.nan)

            df['Occupation'] = df['Occupation'].str.strip()
            df['Occupation'] = df['Occupation'].replace('?', np.nan)

            df['Native Country'] = df['Native Country'].str.strip()
            df['Native Country'] = df['Native Country'].replace('?', np.nan)

            df['Income'] = df['Income'].str.strip()
            df['Income'] = df['Income'].replace({'<=50K' : 0, '>50K' : 1})
            
            # Create Train and Test dataset using train-test-split method
            train_dataset, test_dataset = train_test_split(df, test_size=0.3, random_state=42)

            # Create Train dataset at artifacts folder
            train_dataset.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Create Test dataset at artifacts folder
            test_dataset.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_oath
            )

        except Exception as e:
            raise CustomException(e, sys)