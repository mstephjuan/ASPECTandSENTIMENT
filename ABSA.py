import spacy
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from collections import Counter
#from sentence_transformers import SentenceTransformer
import numpy as np
nlp = spacy.load("en_core_web_lg")
import json
#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#word_model = KeyedVectors.load_word2vec_format("C:\\Users\\kreyg\\OneDrive\\Documents\\word2vec-model\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin", binary=True, limit=500000)

# getAspectDescription(text: string) => [{aspect: string, description: string}]

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

# # TRANSLATED TO ENGLISH
# sentences = [
#     """Effectiveness: long lasting
# Fragrance: long lasting like the original scent
# Fast delivery, only 1 day, deliver immediately, the seller also ships quickly. Smells good and doesn't go away quickly. The smell really clings to the skin. I hope the next time there's a freebie. I'll order again. 5 stars.""",
# """Effectiveness: very effective it last longer
# Fragrance: smell really good
# Texture: Original feels
# recieved the parcel in good condition, great qualitybfor its price!!! thank you shopee,seller qnd the kind courrier as well. god bless you all. thanks for the freebie""",
# """Effectiveness: Stay Longer
# Fragrance: Good
# Texture: Nice
# It smells so bad, thanks seller for your fast shipping. I will order again when it runs out.""",
# """Effectiveness: long lasting, even when I get home at night it still smells good
# Fragrance: the smell is smooth, a great winner for the price
# Texture: the bottle is nice, the former is premium""",
# """The fragrance is really long lasting. You won't regret it, it's worth it. It really smells like original scents. More sales at the net seller. It's still very smart. Thank you.""",
# """Its really good perfume, the smell so good, good nice to smell, the packaging was good. I give 5 stars for this product. Thank you shopee and seller.""",
# """Effectiveness: 10/10
# Fragrance: 10/10
# Amoy all day long. The scent stuck to my clothes""",
# """Fragrance: super scent""",
# ]

def getAspects(sentences):
    aspects = set()  # Use a set instead of a list
    text = ''.join(sentences)
    doc = nlp(text)
    for sent in doc.sents:
        target = []
        for token in sent:
            if (token.dep_ == 'nsubj' or token.dep_ == 'dobj') and token.pos_ == 'NOUN' and token.ent_type_ == '':
                target.append(token.text)
        if target:
            aspects.update(target)  # Use the update() method of set to add elements without duplicates
    return list(aspects) 

def getSentiment(texts):
    model = pickle.load(open("SentimentModel/modelCraig.pkl", 'rb'))
    sentence_sentiments = []
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
            output = "Positive"
    
        sentence_sentiments.append([text, output])

    return sentence_sentiments
    #return output, positive_prob

def mapSentences(sentences):
    aspects = getAspects(sentences)
    #sentiments = getSentiment(sentences)
    groups = {aspect: [] for aspect in aspects}
    for sentence in sentences:
        sentence_aspects = getAspects([sentence])
        sentiment = getSentiment([sentence])
        for aspect in sentence_aspects:
            groups[aspect].append(sentiment)
    return groups

def groupAspects(aspect_list, sentences):
    # Load pre-trained Word2Vec model
    word_model = KeyedVectors.load_word2vec_format("Aspect-Extraction/GoogleNews-vectors-negative300.bin", binary=True, limit=1000000)
    #word_model = KeyedVectors.load_word2vec_format("Aspect-Extraction/GoogleNews-vectors-negative300.bin", binary=True, limit=500000)

    # Convert aspects to word vectors
    aspect_vectors = []
    for aspect in aspect_list:
        if aspect in word_model:
            aspect_vectors.append(word_model[aspect])
        else:
            print(f"Warning: Aspect '{aspect}' not in vocabulary.")

    # Cluster word vectors using k-means
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=7)
    kmeans.fit(aspect_vectors)
    clusters = kmeans.predict(aspect_vectors)

    # Find representative label for each cluster
    labels = []
    grouped_aspects = {}
    for i in range(kmeans.n_clusters):
        cluster_aspects = set()
        for j in range(len(aspect_list)):
            if j < len(clusters) and clusters[j] == i:
                cluster_aspects.add(aspect_list[j])
        aspect_counts = Counter([aspect for sentence in sentences for aspect in cluster_aspects if aspect in sentence])
        most_common_aspect = aspect_counts.most_common(1)[0][0]
        if len(cluster_aspects) >= 2:
            labels.append(most_common_aspect)
            grouped_aspects[most_common_aspect] = cluster_aspects
    return grouped_aspects


def createAspectSentimentDict(groupedAspects, sentenceMaps):
    mapScores = {}
    for aspect_label, aspects in groupedAspects.items():
        mapScores[aspect_label] = {aspect: ", ".join(map(str, sentiment)) for aspect, sentiment in sentenceMaps.items() if aspect in aspects}
    return mapScores

def getOverallSentiment(result):
    new_dict = {}
    for aspect_label, sentiment_dict in result.items():
        positive_sentiment = 0
        negative_sentiment = 0
        for aspect, sentiment in sentiment_dict.items():
            if 'positive' in sentiment.lower():
                positive_sentiment += 1
            elif 'negative' in sentiment.lower():
                negative_sentiment += 1
        total_sentiment = positive_sentiment + negative_sentiment
        if total_sentiment > 0:
            overall_sentiment = positive_sentiment / total_sentiment
        else:
            overall_sentiment = 0
        new_dict[aspect_label] = {'pos-count': positive_sentiment, 'neg-count': negative_sentiment, 'overall-sentiment': overall_sentiment}
    return new_dict

def listSentences(texts):
    text_list = {}
    aspect_sent_dict = createAspectSentimentDict(groupAspects(getAspects(texts), texts), mapSentences(texts))
    for aspect_label, nested_dict in aspect_sent_dict.items():
        pos_list = []
        neg_list = []
        for aspect, sentiment in nested_dict.items():
            if 'positive' in sentiment.lower():
                pos_list.append(sentiment)
            elif 'negative' in sentiment.lower():
                neg_list.append(sentiment)
        text_list[aspect_label] = {'pos': pos_list, 'neg': neg_list}
    return text_list

        

def getABSA(sentences):
    my_aspects = getAspects(sentences)
    my_groupedAspects = groupAspects(my_aspects, sentences)
    my_dict = createAspectSentimentDict(my_groupedAspects, mapSentences(sentences))
    sent_score = getOverallSentiment(my_dict)
    return sent_score

# print(getSentiment(sentences))
print(getABSA(sentences))
print()
print(json.dumps(listSentences(sentences), indent=1))
#print(getAspects(sentences))
# my_aspects = getAspects(sentences)
# my_groupedAspects = groupAspects(my_aspects, sentences)
# group = mapSentences(sentences)
# my_dict = createAspectSentimentDict(groupAspects(my_aspects,sentences), mapSentences(sentences))
# for aspect_label, nested_dict in my_dict.items():
#     print(aspect_label + ":")
#     for aspect, sentiment in nested_dict.items():
#         print("  {} -> \n   {}".format(aspect, sentiment))
#     print()

#print()
#print(json.dumps(getOverallSentiment(my_dict), indent=1))



