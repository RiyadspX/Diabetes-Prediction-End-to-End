import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifact","model.pkl")
            preprocessor_path = os.path.join('artifact','proprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = int(model.predict(data_scaled))
            proba = model.predict_proba(data_scaled)
            return preds, proba[0][1]
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        Pregnancies: float,
        Glucose: float,
        BloodPressure: float,
        SkinThickness: float,
        Insulin: float,
        BMI: float,
        DiabetesPedigreeFunction: float,
        Age: float):
        try:
            self.Pregnancies = Pregnancies
            self.Glucose = Glucose
            self.BloodPressure = BloodPressure
            self.SkinThickness = SkinThickness
            self.Insulin = Insulin
            self.BMI = BMI
            self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
            self.Age = Age
        except Exception as e:
            raise CustomException(e, sys)
        

        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Pregnancies':[self.Pregnancies],
                'Glucose':[self.Glucose],
                'BloodPressure':[self.BloodPressure],
                'SkinThickness':[self.SkinThickness],
                'Insulin':[self.Insulin],
                'BMI':[self.BMI],
                'DiabetesPedigreeFunction':[self.DiabetesPedigreeFunction],
                'Age':[self.Age]
                           
            }
            

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data = CustomData(2, 138, 62, 35, 0, 33.6, 0.127, 47)
    data_df = data.get_data_as_data_frame()
    predictor = PredictPipeline()
    print(predictor.predict(data_df))


