# Important imports
import os
import sys
from dataclasses import dataclass
import importlib

# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# Modelling and Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix, 
    classification_report
)

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# Project helper import 
from src.exception import CustomException
from src.logger import logging
import src.utils
from src.utils import save_object
from src.utils import load_object, evaluate_models



# Create the model trainning config to configure the saved model pkl path
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Splitting train and test data 
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Dictionnary of Models (Function Space F)
            models = {'RandomForest': RandomForestClassifier()
                
                 }

            params = {
            'RandomForest': {
            'n_estimators': [20, 30 , 50, 100, 200],
            'max_depth': [10, 20, 15, 25, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
            }
        }

            # Evaluate models is in the utils file
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                             models=models, param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted=best_model.predict(X_test)

            recall = recall_score(y_test, predicted)
            return recall
                
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__== "__main__":
    print()

