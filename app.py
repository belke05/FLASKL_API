import flask, requests, pickle, joblib
from flask_cors import CORS, cross_origin
import numpy as np

app = flask.Flask(__name__)

# Load the model
model = joblib.load("./linear_regression_model.pkl")
classifier = joblib.load("./classifier.pkl")


cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
def my_index():
  return "<h1>Welcome to our server !!</h1>"


@app.route("/predict_salary", methods=['GET', 'POST'])
def makePrediction():
  if flask.request.method == "GET":
    years_experience = flask.request.args.get('experience')
    years_experience = np.array(np.float(years_experience)).reshape(-1,1)
    print(years_experience, 'years of experience')
    prediction = model.predict(years_experience)
    prediction = prediction.flatten()[0]
    print(prediction, 'predictions')
    return flask.jsonify({'prediction': prediction})
  if flask.request.method == "POST":
    print(flask.request)
    form = flask.request.get_json(force=True)
    print("here")
    formkeys = [key for key in form.keys()]
    formvalues = [np.float(value) for value in form.values()]
    years_experience = formvalues[0]
    years_experience = np.array(np.float(years_experience)).reshape(-1, 1)
    prediction = model.predict(years_experience)
    prediction = prediction.flatten()[0]
    return flask.jsonify({'prediction': prediction})
  

@app.route("/predict_admission", methods=['POST'])
def makeAdmissionPrediction():
  # ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
  form = flask.request.json
  formkeys = [key for key in form.keys()]
  formvalues = [np.float(value) for value in form.values()]
  rndm_vals = np.array(formvalues).reshape(1, -1)
  print(rndm_vals)
  print(form)
  # 337 118 4 4.5 4.5 9.65 1 
  # GRE Score	TOEFL Score	University Rating	SOP	LOR	CGPA	Research
  # change for non admission is 1 change of admission is 0
  prediction = classifier.predict(np.array(rndm_vals))
  prediction = str(prediction.flatten()[0])
  print(prediction, 'predictions')
  return flask.jsonify({'prediction': prediction})



if __name__ == '__main__':
  app.run(debug=True, port=7000)