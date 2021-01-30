from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model

file = 'disaster_prediction_model.pkl'
rclf = pickle.load(open(file, 'rb'))
transformFile = 'transform.pkl'
cv = pickle.load(open(transformFile, 'rb'))
app = Flask(__name__)

@app.route('/', methods=['GET'])
def Home():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  # train_data = pd.read_csv('train.csv')
  # test_data = pd.read_csv('test.csv')
  # count_vectorizer = feature_extraction.text.CountVectorizer()
  # train_vector = count_vectorizer.fit_transform(train_data['text'])
  # rclf = linear_model.RidgeClassifier()
  # rclf.fit(train_vector, train_data['target'])

  if request.method == 'POST':
    text = request.form['tweet']
    data = [text]
    vector =  cv.transform(data).toarray()
    prediction = rclf.predict(vector)
    if prediction == 0:
      return render_template('index.html', prediction_text='Nothing to worry about')
    elif prediction == 1:
      return render_template('index.html', prediction_text='This is an actual disaster. Send help')
  else:
    return render_template('index.html')

if __name__ =="__main__":
  app.run(debug=True)
