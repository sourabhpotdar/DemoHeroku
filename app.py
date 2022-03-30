import pickle

import pandas as pd

from flask import Flask, request, render_template

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [float(i) for i in int_features]
    input_df = pd.DataFrame([final_features], columns=["Weight", "Length1", "Length2", "Length3", "Height", "Width"])
    result = model.predict(input_df)
    return render_template('index.html', prediction_text='The fish belongs to "{}" species.'.format(result[0]))
    

if __name__ == '__main__':
    app.run()
