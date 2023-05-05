import json
from flask import Flask, jsonify, request
import Sentiment as ml

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():

    prediction = ml.getSentiment(["This is bad lol!"])
    print(prediction)
    # Return prediction as JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)