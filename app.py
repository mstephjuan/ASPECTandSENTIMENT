from urllib.parse import unquote
from flask import Flask, jsonify, request, Response
import json
from flask_cors import CORS
from ABSA import (
    getABSA
)

app = Flask(__name__)
cors = CORS(app, resources={r"/data": {"origins": "http://localhost:4200"}})

# Routes and views go here
@app.route('/')
def index():
    return "Hello"

@app.route('/data', methods=['POST'])
def handle_review():
    review_text = request.json.get('reviewText')
    # process the review_text here

    # create a dictionary to return as JSON
    response_data = {'message': review_text}
    print(response_data)

    # return the response as JSON
    return jsonify(response_data)

@app.route('/aspects')

def aspects():
    sentences = [
    # Positive sentences
    "The battery life on this device is impressive.",
    "The camera takes stunning photos in low light.",
    "The screen quality is excellent with vibrant colors.",
    "The performance of this device is incredibly fast.",
    "Battery performance is outstanding, lasting all day.",
    "The camera produces sharp and clear images.",
    "The screen resolution is top-notch and provides a great viewing experience.",
    "This device delivers exceptional performance for demanding tasks.",
    "The battery charges quickly and holds the charge well.",
    "The camera features various modes that enhance photography.",
    "The screen size is perfect, providing ample space for content.",
    "The device handles resource-intensive applications with ease.",
    "Battery efficiency is one of the standout features.",
    "The camera autofocus is quick and accurate.",
    "The screen brightness can be adjusted to suit any environment.",
    
    # Negative sentences
    "The battery drains too quickly and needs frequent charging.",
    "The camera struggles in low light conditions, resulting in blurry photos.",
    "The screen has a noticeable color shift when viewed from certain angles.",
    "The device lags and experiences slowdowns during multitasking.",
    "Battery life is disappointing, requiring constant recharging.",
]

    absa = getABSA(sentences)


 
    return jsonify(absa)


# @app.route('/aspects')
# def aspects():
#     sentences = [
#         # Positive and negative sentences
#         # Add your sentences here
#     ]

#     # Call the necessary functions
#     aspect_list = getAspects(' '.join(sentences))
#     return jsonify(aspect_list)

# @app.route('/sentiments')
# def sentiment():
#     sentences = [
#         # Positive and negative sentences
#         # Add your sentences here
#     ]

#     # Call the necessary functions
#     sentiments = getSentiment(sentences)
#     return jsonify(sentiments)


if __name__ == '__main__':
    app.run(host='localhost', port=5000)