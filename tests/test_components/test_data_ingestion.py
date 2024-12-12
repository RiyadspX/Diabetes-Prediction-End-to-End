import unittest
import os
import pandas as pd
from src.components.data_ingestion import DataIngestion, DataIngestionConfig

class TestDataIngestion(unittest.TestCase):

    def setUp(self):
        """
        Set up temporary configurations and test data for the tests.
        """
        # Create a temporary directory for testing
        self.temp_artifact_dir = "test_artifacts"
        os.makedirs(self.temp_artifact_dir, exist_ok=True)

        # Modify DataIngestionConfig paths to use the temporary directory
        self.ingestion_config = DataIngestionConfig(
            train_data_path=os.path.join(self.temp_artifact_dir, "diabete_train.csv"),
            test_data_path=os.path.join(self.temp_artifact_dir, "diabete_test.csv"),
            raw_data_path=os.path.join(self.temp_artifact_dir, "diabete_data.csv"),
        )

        self.data_ingestion = DataIngestion()
        self.data_ingestion.ingestion_config = self.ingestion_config

        # Create a dummy dataset for testing
        self.dummy_data = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "bmi": [18.5, 22.0, 24.0, 26.5, 30.0],
            "glucose_level": [85, 90, 95, 100, 110],
            "diabetes": [0, 0, 1, 1, 1]
        })
        self.raw_data_path = "notebooks/dataset/kaggle_diabetes.csv"
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        self.dummy_data.to_csv(self.raw_data_path, index=False)

    def tearDown(self):
        """
        Clean up temporary files and directories created during the tests.
        """
        import shutil
        shutil.rmtree(self.temp_artifact_dir, ignore_errors=True)
        os.remove(self.raw_data_path)

    def test_initiate_data_ingestion(self):
        """
        Test the data ingestion process to ensure it correctly reads, splits, 
        and saves the datasets to the configured paths.
        """
        train_path, test_path = self.data_ingestion.initiate_data_ingestion()

        # Check if files are created
        self.assertTrue(os.path.exists(train_path), "Train dataset file was not created.")
        self.assertTrue(os.path.exists(test_path), "Test dataset file was not created.")
        self.assertTrue(os.path.exists(self.ingestion_config.raw_data_path), "Raw dataset file was not created.")

        # Validate the content of the train and test datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        self.assertEqual(len(train_df) + len(test_df), len(self.dummy_data), "Data split size mismatch.")
        
        # Validate if train-test split ratio is correct
        self.assertAlmostEqual(len(test_df) / len(self.dummy_data), 0.2, delta=0.05, msg="Test dataset size is incorrect.")

if __name__ == "__main__":
    unittest.main()
