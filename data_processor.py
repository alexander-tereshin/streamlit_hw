import pickle
import pathlib

import pandas as pd


class MarketingCompanyPredictorPreprocessor:
    def __init__(self, models_folder):
        self.pipeline = self.load_pickle(models_folder, filename='pipeline.pkl')
        self.preprocessor = self.load_pickle(models_folder, filename='preprocessor.pkl')

    @staticmethod
    def load_pickle(folder, filename):
        contents = pickle.load(open((folder / filename), 'rb'))
        return contents

    def make_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = ['EDUCATION', 'MARITAL_STATUS', 'REG_ADDRESS_PROVINCE', 'FAMILY_INCOME']
        num_cols = ['AGE', 'GENDER', 'CHILD_TOTAL', 'DEPENDANTS', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
                    'FL_PRESENCE_FL', 'OWN_AUTO', 'TARGET', 'PERSONAL_INCOME', 'FAMILY_INCOME', 'LOAN_NUM_TOTAL',
                    'LOAN_NUM_CLOSED']

        best_thr = 0.13

        prediction = (self.pipeline.predict_proba(df)[:, 1] > best_thr)[0]
        prediction_proba = self.pipeline.predict_proba(df)[0]

        encode_prediction_proba = {
            0: "Клиент откажется с вероятностью",
            1: "Клиент положительно отреагирует с вероятностью"
        }

        encode_prediction = {
            0: "Не стоит отправлять клиенту предложение о новой услуге.",
            1: "Клиенту стоит попробовать отправить предложение о новой услуге!"
        }

        prediction_data = {}

        for key, value in encode_prediction_proba.items():
            prediction_data.update({value: prediction_proba[key]})

        prediction_df = pd.DataFrame(prediction_data, index=[0])
        prediction_decoded = encode_prediction[prediction]

        return prediction_decoded, prediction_df


