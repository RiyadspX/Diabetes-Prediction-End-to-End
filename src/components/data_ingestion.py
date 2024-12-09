"""
Here we will write our data ingestion code from different sources.

"""

from dataclasses import dataclass
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from data_transformation import DataTransformation
from model_trainer import ModelTrainerConfig, ModelTrainer

"""
Utilité principale de dataclass : Créer des classes simples qui contiennent des données sans écrire beaucoup de code.
Avantage : Automatiquement, Python ajoute des méthodes comme __init__, __repr__, et __eq__ pour cette classe.
"""
# 1. Define the configuration for the Data ingestion component 

@dataclass
class DataIngestionConfig:
    """
    Définit des chemins par défaut pour sauvegarder les données brutes, d'entraînement, et de test 
    Ces fichiers seront sauvegardés dans le dossier appellé artifact.
    """

    train_data_path: str = os.path.join('artifact', "diabete_train.csv")
    test_data_path: str = os.path.join('artifact', "diabete_test.csv")
    raw_data_path: str = os.path.join('artifact', "diabete_data.csv")

# Create the DataIngestion Class

class DataIngestion:
    def __init__(self):
        # Initialise un objet de la classe DataIngestionConfig pour accéder aux chemins configurés
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
            
        """
        Initiates the data ingestion process by reading raw data, splitting it into 
        training and testing datasets, and saving them to specified file paths.

        This function performs the following steps:
        1. Reads a raw CSV file containing the dataset.
        2. Saves the raw data to a specified path.
        3. Splits the dataset into training and testing subsets.
        4. Saves the training and testing datasets to respective paths.

        Returns
        -------
        tuple
            A tuple containing two strings:
            - Path to the training dataset.
            - Path to the testing dataset.

        Raises
        ------
        CustomException
            If an error occurs during data ingestion, it wraps the exception and raises `CustomException`.

        Notes
        -----
        - The paths for saving raw, training, and testing datasets are configured using `self.ingestion_config`.
        - Assumes the raw dataset is available at `notebook/data/kaggle_diabetes.csv`.

        Examples
        --------
        >>> ingestion_config = IngestionConfig(
        ...     raw_data_path="artifact/diabete_data.csv",
        ...     train_data_path="artifact/diabete_train.csv",
        ...     test_data_path="artifact/diabete_test.csv"
        ... )
        >>> data_ingestion = DataIngestion(ingestion_config)
        >>> train_path, test_path = data_ingestion.initiate_data_ingestion()
        >>> print(train_path, test_path)

        """
            
        logging.info("Enterring the Data ingestion method")
        try:
            # Read the csv data into a dataframe
            df = pd.read_csv(r"notebooks\dataset\kaggle_diabetes.csv")
            logging.info("Read the Dataset as Dataframe complete")
            
            # Creer le dossier artifact
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), # os.path.dirname("artifact/train.csv") -> "artifact" only takes dir name
                        exist_ok=True) 
            
            # Sauvegarde des données brutes lues (df) dans le fichier artifact/diabete_data.csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Splitting Train and Test
            logging.info('Splitting into train and test initiated')
            train_set, test_set = train_test_split(df, test_size=.2, random_state=0)

            # Sauvegarde des données train et test dans le fichier artifact/diabete_strain.csv et artifact/diabete_test.csv resp
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("The Ingestion of Data is completed")

            # On retourne le path du train et du test.
            return ( 
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=="__main__":
    obj = DataIngestion()

    transf = DataTransformation()
    modeltrainer = ModelTrainer()

    train_data_path, test_data_path = obj.initiate_data_ingestion()
    train_arr, test_arr, preprocessor_obj_file_path = transf.initiate_data_transformation(train_data_path, test_data_path) 
    recall = modeltrainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
    print(recall)

            

