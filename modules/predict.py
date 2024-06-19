import pandas as pd
import dill
import os
import json


def load_model():
    for file in os.listdir('../airflow_hw/data/models'):
        if file.endswith('.pkl'):
            with open('../airflow_hw/data/models/' + file, 'rb') as f:
                model = dill.load(f)
                return model


def collect_preds():
    predictions = pd.DataFrame(columns=['id', 'prediction'])  # Initialize empty DataFrame with columns
    model = load_model()
    file_path = '../airflow_hw/data/test/'
    for file in os.listdir(file_path):
        file_loc = file_path + file
        with open(file_loc, 'r') as f:
            json_file = f.read()
        data = json.loads(json_file)
        df = pd.DataFrame.from_dict(data, orient='index').T
        pred = model.predict(df)
        predictions = pd.concat([predictions, pd.DataFrame({'id': [data['id']], 'prediction': pred})], ignore_index=True)
    return predictions


def predict():
    return collect_preds().to_csv('../airflow_hw/data/predictions/predictions.csv', index=False)


if __name__ == '__main__':
    predict()
