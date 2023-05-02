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
    model = pickle.load(open('modelNB.pkl', 'rb'))

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
        review_text = BeautifulSoup(text).get_text()

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

def groupAspects(aspect_list):
    # Load pre-trained Word2Vec model
    word_model = KeyedVectors.load_word2vec_format('Aspect-Extraction/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

    # Convert aspects to word vectors
    aspect_vectors = [word_model[aspect] for aspect in aspect_list]

    # Cluster word vectors using k-means
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(aspect_vectors)
    clusters = kmeans.predict(aspect_vectors)

    # Find representative label for each cluster
    labels = []
    grouped_aspects = {}
    for i in range(kmeans.n_clusters):
        cluster_aspects = [aspect_list[j] for j in range(len(aspect_list)) if clusters[j] == i]
        most_common_aspect = Counter(cluster_aspects).most_common(1)[0][0]
        labels.append(most_common_aspect)
        grouped_aspects[most_common_aspect] = cluster_aspects

    return grouped_aspects
from sklearn.metrics.pairwise import cosine_similarity

def group_sentiments(aspect_sentiments, grouped_aspects, embedder):
    result = {}
    used_sentences = set()
    for label, aspects in grouped_aspects.items():
        result[label] = []
        aspect_embeddings = embedder(aspects)
        for text, output, positive_prob in aspect_sentiments:
            if text not in used_sentences:
                text_embedding = embedder([text])[0]
                similarities = cosine_similarity(text_embedding.reshape(1,-1), aspect_embeddings)
                if max(similarities[0]) > 0.33:
                    result[label].append((text, output, positive_prob))
                    used_sentences.add(text)
    return result

def embedder(texts):
    return embed(texts).numpy()

sentences = [
    "The device has a large and vibrant display that makes everything look great.",
    "Its camera takes stunning photos with vivid colors and sharp details.",
    "The battery life is impressive and lasts all day with heavy use.",
    "The size is perfect for one-handed use and fits comfortably in a pocket.",
    "The screen is responsive and easy to navigate with intuitive gestures.",
    "The pictures captured by the camera are of professional quality.",
    "The life of the device is extended by its durable construction and regular software updates.",
    "The colors on the display are accurate and true to life.",
    "The performance is smooth and fast, even when running multiple apps at once.",
    "The design is sleek and modern, with a premium feel.",
    "The display is protected by scratch-resistant glass for added durability.",
    "The camera has advanced features such as portrait mode and night mode for stunning photos in any lighting condition.",
    "The battery charges quickly and supports wireless charging for added convenience.",
    "The size of the device is perfect for watching videos and playing games.",
    "The screen has a high resolution for sharp and clear images."
]

new_text = ' '.join(sentences)
aspect_list = getAspects(new_text)
group_aspects = groupAspects(aspect_list)
#print(group_aspects)

sentiments = getSentiment(sentences)
print(embedder(sentences))
group_sentences = group_sentiments(sentiments, group_aspects, embedder)
for label, sentences in group_sentences.items():
    print(f"{label}:")
    for sentence in sentences:
        print(f"  {sentence}")

#print(group_aspects)
#for item in group_sentences:
    #print(item)
#for sentiment in getSentiment(sentences):
    #positive_prob = sentiment[2]
    #print(type(positive_prob))
