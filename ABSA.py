import spacy
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import tensorflow_hub as hub
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
import numpy as np
nlp = spacy.load("en_core_web_lg")
import json
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# getAspectDescription(text: string) => [{aspect: string, description: string}]

def getAspects(text):
    aspects = []
    doc = nlp(text)
    for sent in doc.sents:
        target = []
        for token in sent:
            if (token.dep_ == 'nsubj' or token.dep_ == 'dobj') and token.pos_ == 'NOUN' and token.ent_type_ == '':
                target.append(token.text)
        if target:
            aspects.extend(target)
    return aspects

def getSentiment(texts):
    model = pickle.load(open("SentimentModel/bigdata2modelNB.pkl", 'rb'))

    aspect_sentiments = []
    for text in texts:
        """
        function getSentiment(raw_text: string) -> (output: string, prediction: (polarity, subjectivity))

        This function takes in a string of raw text and performs sentiment analysis to determine whether the text is positive or negative. It returns a tuple consisting of the sentiment label and the positive probability of the prediction.

        Args:
            raw_text (str): The raw text to analyze.

        Returns:
            tuple: A tuple consisting of the sentiment label and the positive probability of the prediction.

        Example:
            >>> raw_text = "This product is amazing! I love it so much."
            >>> getSentiment(raw_text)
            ('Positive', 0.00819811, 0.99180189))
        """

        # Instantiate PorterStemmer
        p_stemmer = PorterStemmer()

        # Remove HTML
        review_text = BeautifulSoup(text, features="html.parser").get_text()

        # Remove non-letters
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)

        # Convert words to lower case and split each word up
        words = letters_only.lower().split()

        # Convert stopwords to a set
        stops = set(stopwords.words('english'))

        # Adding on stopwords that were appearing frequently in both positive and negative reviews
        stops.update(['app','shopee','shoppee','item','items','seller','sellers','bad'])

        # Remove stopwords
        meaningful_words = [w for w in words if w not in stops]

        # Stem words
        meaningful_words = [p_stemmer.stem(w) for w in meaningful_words]

        # Join words back into one string, with a space in between each word
        final_text = pd.Series(" ".join(meaningful_words))

        # Generate predictions
        pred = model.predict(final_text)[0]
        positive_prob = model.predict_proba([pd.Series.to_string(final_text)])[0][0]


        if pred == 1:
            output = "Negative"
        else:
            output = "Postive"
    
        aspect_sentiments.append([text, output, positive_prob])

    return aspect_sentiments
    #return output, positive_prob


from collections import Counter

def groupAspects(aspect_list, sentences):
    
    word_model = KeyedVectors.load_word2vec_format("Aspect-Extraction/GoogleNews-vectors-negative300.bin", binary=True, limit=500000)

    # Convert aspects to word vectors
    aspect_vectors = [word_model[aspect] for aspect in aspect_list]

    # Cluster word vectors using k-means
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(aspect_vectors)
    clusters = kmeans.predict(aspect_vectors)

    # Find representative label for each cluster
    labels = []
    grouped_aspects = {}
    #used_aspects = set()
    for i in range(kmeans.n_clusters):
        cluster_aspects = set(aspect_list[j] for j in range(len(aspect_list)) if clusters[j] == i)
        aspect_counts = Counter([aspect for sentence in sentences for aspect in cluster_aspects if aspect in sentence])
        most_common_aspect = aspect_counts.most_common(1)[0][0]
        labels.append(most_common_aspect)
        grouped_aspects[most_common_aspect] = cluster_aspects
    return grouped_aspects

from sklearn.metrics.pairwise import cosine_similarity

def group_sentiments(aspect_sentiments, grouped_aspects, embedder):
    result = {}
    used_sentences = set()
    for label in grouped_aspects.keys():
        result[label] = []
        label_embedding = embedder([label])[0]
        for text, output, positive_prob in aspect_sentiments:
            if text not in used_sentences:
                text_embedding = embedder([text])[0]
                similarity = cosine_similarity(text_embedding.reshape(1,-1), label_embedding.reshape(1,-1))
                if similarity > 0.14:
                    result[label].append((text, output, positive_prob))
                    used_sentences.add(text)
    return result

def extract_positive_probabilities(result):
    new_dict = {}
    for label, values in result.items():
        scores = [positive_prob for text, output, positive_prob in values]
        if len(scores) > 0:
            sentiment_score = sum(scores) / len(scores)
        else:
            sentiment_score = 0
        new_dict[label] = sentiment_score * 100
    return new_dict
def embedder(texts):
    return embed(texts).numpy()

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

new_text = ' '.join(sentences)
aspect_list = getAspects(new_text)
group_aspects = groupAspects(aspect_list, sentences)
#print(group_aspects)

sentiments = getSentiment(sentences)
#print(embedder(sentences))
group_sentences = group_sentiments(sentiments, group_aspects, embedder)
# for label, sentences in group_sentences.items():
#     print(f"{label}:")
#     for sentence in sentences:
#         print(f"  {sentence}")
# overall_sent_score = extract_positive_probabilities(group_sentences)

if __name__ == '__main__':
    overall_sent_score = extract_positive_probabilities(group_sentences)
    output_json = json.dumps(overall_sent_score)
    print(output_json)

#for item in group_sentences:
    #print(item)
#for sentiment in getSentiment(sentences):
    #positive_prob = sentiment[2]
    #print(type(positive_prob))
