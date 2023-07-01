import asyncio
from urllib.parse import unquote
from flask import Flask, jsonify, request, Response, send_from_directory
import json
from flask_cors import CORS
from gensim.models import KeyedVectors
from final_absa import (
    ExtractAspects,
    ExtractTopAspects,
    ExtractAspectPhrases,
    getRawSentimentScore,
    getNormalizedSentimentScore,
    analyzeAspectPhrases,
    analyzeAllReviews
)
from scraper import (
    scrape,
    getProductTitle
)
import pickle

model = pickle.load(open("SentimentModel/modelCraig.pkl", 'rb'))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/absa-dashboard', methods=['POST'])
def absa_dashboard():
    if request.method == 'OPTIONS':
        return Response(status=200)
    if request.method == 'POST':
        data = request.json
        print(data)
        url = data['url']
        print(url)
        if url == 'https://www.amazon.com/Nikon-COOLPIX-P1000-Digital-Camera/product-reviews/B07F5HPXK4/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews':
            with open('nikon_COOLPIX.json', 'r') as f:
                reviews = json.load(f)
            title = getProductTitle(url)
            aspects = ExtractAspects(reviews)
            # print(aspects)
            top_aspects = ExtractTopAspects(reviews, aspects)
            aspect_phrases = ExtractAspectPhrases(reviews, top_aspects)
            raw_score = getRawSentimentScore(aspect_phrases)
            normalized_score = getNormalizedSentimentScore(aspect_phrases)
            phrases_analysis = analyzeAspectPhrases(aspect_phrases)
            reviews_analysis = analyzeAllReviews(reviews)
            output = {
                "title": title,
                "aspects": aspects,
                "top_aspects": top_aspects,
                # "aspect_phrases": aspect_phrases,
                "raw_score": raw_score,
                "normalized_score": normalized_score,
                "phrases_analysis": phrases_analysis,
                "reviews_analysis": reviews_analysis
            }
            # print(output)
            return json.dumps(output)
        elif url == 'https://www.amazon.com/Sanabul-Womens-Easter-Boxing-Gloves/product-reviews/B08L87WGF4/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1':
            with open('purple_gloves.json', 'r') as f:
                reviews = json.load(f)
            title = getProductTitle(url)
            aspects = ExtractAspects(reviews)
            # print(aspects)
            top_aspects = ExtractTopAspects(reviews, aspects)
            aspect_phrases = ExtractAspectPhrases(reviews, top_aspects)
            raw_score = getRawSentimentScore(aspect_phrases)
            normalized_score = getNormalizedSentimentScore(aspect_phrases)
            phrases_analysis = analyzeAspectPhrases(aspect_phrases)
            reviews_analysis = analyzeAllReviews(reviews)
            output = {
                "title": title,
                "aspects": aspects,
                "top_aspects": top_aspects,
                # "aspect_phrases": aspect_phrases,
                "raw_score": raw_score,
                "normalized_score": normalized_score,
                "phrases_analysis": phrases_analysis,
                "reviews_analysis": reviews_analysis
            }
            # print(output)
            return json.dumps(output)
        elif url:
            title = getProductTitle(url)
            reviews = scrape(url)
            aspects = ExtractAspects(reviews)
            # print(aspects)
            top_aspects = ExtractTopAspects(reviews, aspects)
            aspect_phrases = ExtractAspectPhrases(reviews, top_aspects)
            raw_score = getRawSentimentScore(aspect_phrases)
            normalized_score = getNormalizedSentimentScore(aspect_phrases)
            phrases_analysis = analyzeAspectPhrases(aspect_phrases)
            reviews_analysis = analyzeAllReviews(reviews)
            output = {
                "title": title,
                "aspects": aspects,
                "top_aspects": top_aspects,
                # "aspect_phrases": aspect_phrases,
                "raw_score": raw_score,
                "normalized_score": normalized_score,
                "phrases_analysis": phrases_analysis,
                "reviews_analysis": reviews_analysis
            }
            # print(output)
            return json.dumps(output)
        else:
            return 'URL does not exist', 404
            
    return Response(status=400)

if __name__ == '__main__':
    app.run(host='localhost', port=8080, threaded=False)
