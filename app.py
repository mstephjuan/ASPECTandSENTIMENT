from urllib.parse import unquote
from flask import Flask, jsonify, request, Response, send_from_directory
import json
from flask_cors import CORS
from ABSA import (
    createAspectSentimentDict,
    getABSA,
    getSentiment,
    listSentences as processSentences,  # Renamed the imported function,
    getAspects,
    groupAspects,
    mapSentences,
    getListSentences,
    getCountSentiments,
    countSentiments,
    sentenceAttributes,
    visualizeAspectSentiment
)

app = Flask(__name__)
# Allow requests from http://localhost:4200 and http://localhost:3000 to all endpoints in your Flask app. You can customize this according to your needs.
# Allow all origins for now

CORS(app, resources={r"/*": {"origins": "*"}})
# CORS(app, resources={r"/*": {"origins": ["http://localhost:4200", "http://localhost:3000"]}})

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

    # return the response as JSON
    return jsonify(response_data)

@app.route('/aspects')
def aspects():
    """
    This API endpoint returns the aspect-based sentiment analysis (ABSA) of a list of sentences.
    
    :return: A JSON object containing the ABSA of the sentences.
    :rtype: dict
    """
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

    # Sample output of getABSA
    # {
    #     "device": {
    #         "neg-count": 2,
    #         "overall-sentiment": 0.75,
    #         "pos-count": 6
    #     },
    #     "life": {
    #         "neg-count": 5,
    #         "overall-sentiment": 0.5,
    #         "pos-count": 5
    #     },
    #     "performance": {
    #         "neg-count": 2,
    #         "overall-sentiment": 0.6666666666666666,
    #         "pos-count": 4
    #     }
    # }
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


# Just like the aspects route, but with a POST request that accepts a JSON body
@app.route('/extract-aspects', methods=['POST'])
def extract_aspects():
    '''
    Extract aspects from a list of sentences
    
    API Specification:
    Format: {"sentences": ["sentence1", "sentence2", ...]}
    Example Input: {"sentences": ["The battery life on this device is impressive.", "The camera takes stunning photos in low light."]}
    
    Parameters:
        JSON (str): A JSON object with a list of sentences

    Returns:
        JSON (str): A JSON object with a summary of the aspects and sentiment score

    Sample output:
    {
        "device": {
            "neg-count": 2,
            "overall-sentiment": 0.75,
            "pos-count": 6
        },
        "life": {
            "neg-count": 5,
            "overall-sentiment": 0.5,
            "pos-count": 5
        },
        "performance": {
            "neg-count": 2,
            "overall-sentiment": 0.6666666666666666,
            "pos-count": 4
        }
    }
    '''
    # OPTIONS request is sent by the browser to check if the server allows the request
    # If the server does not allow the request, the browser will not send the POST request
    # This is a CORS (Cross-Origin Resource Sharing) preflight request
    # Accept json type
    if request.method == 'OPTIONS':
        print("OPTIONS accessed")
        return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'
    
    if request.method == 'POST':
        data = request.json
        sentences = data['sentences']
        # absa = getABSA(sentences)
        count = getCountSentiments(sentences)
        return jsonify(count)
    
    return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'


@app.route('/sentiment', methods=['POST'])
def sentiment():

    # OPTIONS request is sent by the browser to check if the server allows the request
    # If the server does not allow the request, the browser will not send the POST request
    # This is a CORS (Cross-Origin Resource Sharing) preflight request
    # Accept json type
    if request.method == 'OPTIONS':
        print("OPTIONS accessed")
        return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'
    
    if request.method == 'POST':
        data = request.json
        sentences = data['sentence']
        # Sample output of getABSA
        # {'screen': {'environment', 'charging', 'experience', 'applications', 'space', 'screen', 'efficiency', 'life', 'modes', 'charge', 'resolution', 'quality', 'shift', 'tasks', 'size', 'slowdowns', 'performance'}, 'camera': {'photos', 'camera', 'photography', 'images'}, 'device': {'recharging', 'device', 'battery'}}
        # {'aspect-group1': {aspect1, aspect2, ...}, 'aspect-group2': {aspect3, aspect4, ...}, ...}
        sentiment = getSentiment(sentences)

        return json.dumps(sentiment)
    
    return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'

@app.route('/absa-extract', methods=['POST'])
def absa_extract():

    # OPTIONS request is sent by the browser to check if the server allows the request
    # If the server does not allow the request, the browser will not send the POST request
    # This is a CORS (Cross-Origin Resource Sharing) preflight request
    # Accept json type
    if request.method == 'OPTIONS':
        print("OPTIONS accessed")
        return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'
    
    if request.method == 'POST':
        data = request.json
        sentences = data['sentences']

        result = {}

        my_aspects = getAspects(sentences)
        my_groupedAspects = groupAspects(my_aspects, sentences)
        group = mapSentences(sentences)
        my_dict = createAspectSentimentDict(groupAspects(my_aspects,sentences), mapSentences(sentences))
        for aspect_label, nested_dict in my_dict.items():
            result[aspect_label] = {}
            for aspect, sentiment in nested_dict.items():
                # insert aspect and sentiment to the aspect_label key in result dictionary
                result[aspect_label][aspect] = sentiment

        return json.dumps(result)
    
    return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'


@app.route('/get-aspects', methods=['POST'])
def get_aspects():

    # OPTIONS request is sent by the browser to check if the server allows the request
    # If the server does not allow the request, the browser will not send the POST request
    # This is a CORS (Cross-Origin Resource Sharing) preflight request
    # Accept json type
    if request.method == 'OPTIONS':
        print("OPTIONS accessed")
        return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'
    
    if request.method == 'POST':
        data = request.json
        sentences = data['sentences']
        result = getAspects(sentences)

        return json.dumps(result)
    
    return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'



@app.route('/map-sentences', methods=['POST'])
def map_sentences():

    # OPTIONS request is sent by the browser to check if the server allows the request
    # If the server does not allow the request, the browser will not send the POST request
    # This is a CORS (Cross-Origin Resource Sharing) preflight request
    # Accept json type
    if request.method == 'OPTIONS':
        print("OPTIONS accessed")
        return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'
    
    if request.method == 'POST':
        data = request.json
        sentences = data['sentences']
        result = mapSentences(sentences)

        return json.dumps(result)
    
    return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'



@app.route('/group-aspects', methods=['POST'])
def group_aspects():

    # OPTIONS request is sent by the browser to check if the server allows the request
    # If the server does not allow the request, the browser will not send the POST request
    # This is a CORS (Cross-Origin Resource Sharing) preflight request
    # Accept json type
    if request.method == 'OPTIONS':
        print("OPTIONS accessed")
        return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'
    
    if request.method == 'POST':
        data = request.json
        sentences = data['sentences']
        aspect_list = data['aspect_list']
        result = groupAspects(aspect_list, sentences)

        # Convert the values of the dictionary to list
        for key, value in result.items():
            result[key] = list(value)

        return json.dumps(result)
    
    return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'


@app.route('/list-sentences', methods=['POST'])
def list_sentences():
    # OPTIONS request is sent by the browser to check if the server allows the request
    # If the server does not allow the request, the browser will not send the POST request
    # This is a CORS (Cross-Origin Resource Sharing) preflight request
    # Accept json type
    if request.method == 'OPTIONS':
        print("OPTIONS accessed")
        return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'
    
    if request.method == 'POST':
        data = request.json
        sentences = data['sentences']
        result = getListSentences(sentences)

        return json.dumps(result)
    
    return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'


@app.route('/count-sentiments', methods=['POST'])
def count_sentiments():
    # OPTIONS request is sent by the browser to check if the server allows the request
    # If the server does not allow the request, the browser will not send the POST request
    # This is a CORS (Cross-Origin Resource Sharing) preflight request
    # Accept json type
    if request.method == 'OPTIONS':
        print("OPTIONS accessed")
        return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'
    
    if request.method == 'POST':
        data = request.json
        sentences = data['sentences']
        result = getCountSentiments(sentences)

        return json.dumps(result)
    
    return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'

@app.route('/visualize', methods=['POST'])
def visualize():
    # OPTIONS request is sent by the browser to check if the server allows the request
    # If the server does not allow the request, the browser will not send the POST request
    # This is a CORS (Cross-Origin Resource Sharing) preflight request
    # Accept json type
    if request.method == 'OPTIONS':
        print("OPTIONS accessed")
        return 'This is the Visualize endpoint. Send a POST request with a list of sentences to visualize.'
    
    if request.method == 'POST':
        data = request.json
        sentences = data['sentences']
        absa = getABSA(sentences)
        result = visualizeAspectSentiment(absa)

        return json.dumps(result)
    
    return 'This is the Visualize endpoint. Send a POST request with a list of sentences to visualize.'

# Combined for optimization
@app.route('/init-dashboard', methods=['POST'])
def init_dashboard():
    # OPTIONS request is sent by the browser to check if the server allows the request
    # If the server does not allow the request, the browser will not send the POST request
    # This is a CORS (Cross-Origin Resource Sharing) preflight request
    # Accept json type
    if request.method == 'OPTIONS':
        print("OPTIONS accessed")
        return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'
    
    if request.method == 'POST':
        data = request.json
        sentences = data['sentences']

        my_aspects = getAspects(sentences)
        my_groupedAspects = groupAspects(my_aspects, sentences)


        my_dict = createAspectSentimentDict(my_groupedAspects, mapSentences(sentences))
        get_list_sentences = processSentences(mapSentences(sentences), my_groupedAspects)
        get_count_sentiments = countSentiments(mapSentences(sentences), my_groupedAspects)
        sentence_attributes = sentenceAttributes(sentences, my_groupedAspects)

        result = {
            "get_absa": my_dict,
            "get_list_sentences": get_list_sentences,
            "get_count_sentiments": get_count_sentiments,
            "get_aspect_groups": my_groupedAspects,
            "sentence_attributes": sentence_attributes
        }

        return json.dumps(result)
    
    return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'


@app.route('/sentence-attributes', methods=['POST'])
def sentence_attributes():
    # OPTIONS request is sent by the browser to check if the server allows the request
    # If the server does not allow the request, the browser will not send the POST request
    # This is a CORS (Cross-Origin Resource Sharing) preflight request
    # Accept json type
    if request.method == 'OPTIONS':
        print("OPTIONS accessed")
        return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'
    
    if request.method == 'POST':
        data = request.json
        sentences = data['sentences']
        aspect_list = getAspects(sentences)
        grouped_aspects = groupAspects(aspect_list, sentences) # structure: {main_aspect: [sub_aspects]}
        result = sentenceAttributes(sentences, grouped_aspects)

        return json.dumps(result)
    
    return 'This is the Extract Aspects endpoint. Send a POST request with a list of sentences to extract aspects.'


# Serve files from the static directory
# /static/templates/index.html
@app.route('/static/templates/<path:path>')
def serve_static(path):
    return send_from_directory('static/templates', path)


if __name__ == '__main__':
    app.run(host='localhost', port=5000)