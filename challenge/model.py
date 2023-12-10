import warnings
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import logging
import os

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.preprocessing')

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")


labels_OPERA = [
    "American Airlines", "Air Canada", "Air France", "Aeromexico",
    "Aerolineas Argentinas", "Austral", "Avianca", "Alitalia",
    "British Airways", "Copa Air", "Delta Air", "Gol Trans", "Iberia",
    "K.L.M.", "Qantas Airways", "United Airlines", "Grupo LATAM",
    "Sky Airline", "Latin American Wings", "Plus Ultra Lineas Aereas",
    "JetSmart SPA", "Oceanair Linhas Aereas", "Lacsa"
]

labels_MES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

labels_TIPOVUELO = ['I', 'N']

column_order = ['OPERA_Aerolineas Argentinas', 'OPERA_Aeromexico', 'OPERA_Air Canada',
       'OPERA_Air France', 'OPERA_Alitalia', 'OPERA_American Airlines',
       'OPERA_Austral', 'OPERA_Avianca', 'OPERA_British Airways',
       'OPERA_Copa Air', 'OPERA_Delta Air', 'OPERA_Gol Trans',
       'OPERA_Grupo LATAM', 'OPERA_Iberia', 'OPERA_JetSmart SPA',
       'OPERA_K.L.M.', 'OPERA_Lacsa', 'OPERA_Latin American Wings',
       'OPERA_Oceanair Linhas Aereas', 'OPERA_Plus Ultra Lineas Aereas',
       'OPERA_Qantas Airways', 'OPERA_Sky Airline', 'OPERA_United Airlines',
       'TIPOVUELO_I', 'TIPOVUELO_N', 'MES_1', 'MES_2', 'MES_3', 'MES_4',
       'MES_5', 'MES_6', 'MES_7', 'MES_8', 'MES_9', 'MES_10', 'MES_11',
       'MES_12']


class DelayModel:

    def __init__(self):
        self._model = None # Model should be saved in this attribute.

    def get_rate_from_column(self, data, column):
        delays = {}
        for _, row in data.iterrows():
            if row["delay"] == 1:
                if row[column] not in delays:
                    delays[row[column]] = 1
                else:
                    delays[row[column]] += 1
        total = data[column].value_counts().to_dict()
        
        rates = {}
        for name, total in total.items():
            if name in delays:
                rates[name] = round(total / delays[name], 2)
            else:
                rates[name] = 0
                
        return pd.DataFrame.from_dict(data = rates, orient = "index", columns = ["Tasa (%)"])


    # Period of day classifier
    def get_period_day(self, date):
        date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
        morning_min = datetime.strptime("05:00", "%H:%M").time()
        morning_max = datetime.strptime("11:59", "%H:%M").time()
        afternoon_min = datetime.strptime("12:00", "%H:%M").time()
        afternoon_max = datetime.strptime("18:59", "%H:%M").time()
        evening_min = datetime.strptime("19:00", "%H:%M").time()
        evening_max = datetime.strptime("23:59", "%H:%M").time()
        night_min = datetime.strptime("00:00", "%H:%M").time()
        night_max = datetime.strptime("4:59", "%H:%M").time()
        
        if(date_time > morning_min and date_time < morning_max):
            return "mañana"
        elif(date_time > afternoon_min and date_time < afternoon_max):
            return "tarde"
        elif(
            (date_time > evening_min and date_time < evening_max) or
            (date_time > night_min and date_time < night_max)
        ):
            return "noche"
        

    # Is High Season classifier
    def is_high_season(self, fecha):
        fecha_año = int(fecha.split("-")[0])
        fecha = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
        range1_min = datetime.strptime("15-Dec", "%d-%b").replace(year = fecha_año)
        range1_max = datetime.strptime("31-Dec", "%d-%b").replace(year = fecha_año)
        range2_min = datetime.strptime("1-Jan", "%d-%b").replace(year = fecha_año)
        range2_max = datetime.strptime("3-Mar", "%d-%b").replace(year = fecha_año)
        range3_min = datetime.strptime("15-Jul", "%d-%b").replace(year = fecha_año)
        range3_max = datetime.strptime("31-Jul", "%d-%b").replace(year = fecha_año)
        range4_min = datetime.strptime("11-Sep", "%d-%b").replace(year = fecha_año)
        range4_max = datetime.strptime("30-Sep", "%d-%b").replace(year = fecha_año)
        
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0
        
    # Get delay in minutes
    def get_min_diff(self, data):
        fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff    
        

    def preprocess(self,data: pd.DataFrame,target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        try:

            # Create separate encoders for each categorical column
            encoder_opera = OneHotEncoder(handle_unknown='error', sparse=False)
            encoder_tipovuelo = OneHotEncoder(handle_unknown='error', sparse=False)
            encoder_mes = OneHotEncoder(handle_unknown='error', sparse=False)

            # Fit the encoders
            encoder_opera.fit(np.array(labels_OPERA).reshape(-1, 1))
            encoder_tipovuelo.fit(np.array(labels_TIPOVUELO).reshape(-1, 1))
            encoder_mes.fit(np.array(labels_MES).reshape(-1, 1))          


            self._encoder = {
                "OPERA": encoder_opera,
                "TIPOVUELO": encoder_tipovuelo,
                "MES": encoder_mes
            }

            features = pd.DataFrame()

            # Apply the fitted encoder to the data
            for column in ["OPERA", "TIPOVUELO", "MES"]:

                encoded_values = self._encoder[column].transform(data[column].values.reshape(-1, 1))
                encoded_df = pd.DataFrame(encoded_values, columns=self._encoder[column].get_feature_names_out([column]))
                features = pd.concat([features, encoded_df], axis=1)

            features = features[column_order]

            if target_column is not None:

                logging.info("Target column provided: proceeding with training preprocessing.")
                # 1. Features Generation
                data["min_diff"] = data.apply(self.get_min_diff, axis=1)

                threshold_in_minutes = 15
                data[target_column] = np.where(data["min_diff"] > threshold_in_minutes, 1, 0)

                return features, pd.DataFrame(data[target_column])


            logging.info("Target column not provided: proceeding with prediction preprocessing.")
            return features
    
        except Exception as e:
            return {"error": str(e)}


    def fit(self, features: pd.DataFrame, target: pd.DataFrame, test_size_input=0.33, export_flag="", model_name="") -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
            test_size(float): test_size.
            export_flag(str): if = "export", export x_train,x_test,y_train,y_test
        """

        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = test_size_input, random_state = 42)
        self._model = LogisticRegression(max_iter=1000)
        self._model.fit(x_train, y_train)
        reg_y_preds = self._model.predict(x_test)

        print(classification_report(y_test, reg_y_preds))

        if model_name:
            joblib.dump(self._model,f"model_{model_name}.mdl")

        if export_flag == "export":
            return x_train, x_test, y_train, y_test
        

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            raise ValueError("Model not fitted. Please fit the model using the 'fit' method.")

        return self._model.predict(features).tolist()
    
    def load(self, model_file_name: str):
        """
        Load persisted model.

        Args:
            model

        Returns:
            (str): confirmation message.
        """

        try:

            self._model = joblib.load(model_file_name)
            return logging.info("Model loaded succesfully.")

        except:
            logging.error("Model file not found.")
