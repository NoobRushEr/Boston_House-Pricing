import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import warnings

# Filter out UserWarnings from sklearn
warnings.filterwarnings("ignore", category=UserWarning)


app=Flask(__name__)
# Load the Model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Receive JSON data from the request
    data = request.json['data']

    # Reshape the data into the format expected by the model
    data_array = np.array(list(data.values())).reshape(1, -1)

    # Make prediction using the trained model
    output = regmodel.predict(scalar.transform(data_array))

    # Convert ndarray to a Python list
    output_list = output.tolist()

    # Return the prediction result as JSON
    return jsonify(output_list)


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



if __name__ == "__main__":
    app.run(debug=False)