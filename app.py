from flask import Flask, jsonify
import json
from ABSA import (
    getAspects,
    getSentiment,
    groupAspects,
    group_sentiments,
    extract_positive_probabilities,
    embedder,
)

app = Flask(__name__)

# Routes and views go here
@app.route('/')
def index():
    sentences = [
        # Positive and negative sentences
        # Add your sentences here
    ]

    # Call the necessary functions
    aspect_list = getAspects(' '.join(sentences))
    group_aspects = groupAspects(aspect_list, sentences)
    sentiments = getSentiment(sentences)
    group_sentences = group_sentiments(sentiments, group_aspects, embedder)
    overall_sent_score = extract_positive_probabilities(group_sentences)
    output_json = json.dumps(overall_sent_score)

    return output_json

@app.route('/aspects')
def aspects():
    sentences = [
        # Positive and negative sentences
        # Add your sentences here
    ]

    # Call the necessary functions
    aspect_list = getAspects(' '.join(sentences))
    return jsonify(aspect_list)

@app.route('/sentiments')
def sentiment():
    sentences = [
        # Positive and negative sentences
        # Add your sentences here
    ]

    # Call the necessary functions
    sentiments = getSentiment(sentences)
    return jsonify(sentiments)


if __name__ == '__main__':
    app.run(debug=True)