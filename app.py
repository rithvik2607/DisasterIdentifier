from flask import Flask, render_template, request
import jsonify
import requests
import pickle
from sklearn import feature_extraction

app = Flask(__name__)
model = pickle.load(open('disaster_prediction_model.pkl', 'rb'))
@app.route('/', methods=['GET'])
def Home():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    text = request.form['tweet']
    count_vectorizer = feature_extraction.text.CountVectorizer()
    vectors = count_vectorizer.transform(text)
    prediction = model.predict(vectors)
    if prediction == 0:
      return render_template('index.html', prediction_text='Nothing to worry about')
    elif prediction == 1:
      return render_template('index.html', prediction_text='This is an actual disaster. Send help')
  else:
    return render_template('index.html')

if __name__ =="__main__":
  app.run(debug=True)
