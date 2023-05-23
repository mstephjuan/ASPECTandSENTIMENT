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
    # Mixed positive and negative
    "It works well, but sometimes the unit just turns off while in use for no reason. I make sure the unit is charged so its not that. The sound quality is fair but nothing to get excited about. ",
    "The product seems to work as intended and is operating very well but it got a bit glitchy when it was first used but after a couple of minutes it worked pretty much fine after that.",
    "the package is ok but the item it self didn't really meet my expectation, there's a glitching sounds I heard while playing a music and I don't know if the earphones were built like this or somethin",
    "Works fine. The sound quality is okay. But the battery life is not that good in terms of talk time. At 100'%' charge, it always runs out of battery at the end of my hour-long meetings.",
    "fast delivery, but the both earphones has different volume, kinda fast to get empty, i love the design, but my biggest problem is when i try to close it ig the magnet is not the strong, overall its fine...",
    "I ordered it yesterday and received it today. it got wet from the rain but thankfully it didn't damage the item. the touch activated sensors do not work but maybe it's because I'm using it on an Android",
    "It's been 2 weeks since I started using this. Sound is good. Audio is clear. And I was impressed by the durability. Because I have accidentally dropped this 3x already yet until now it still functions really well. And I love that! Thank you!",
    "Delivery was so fast, I even thought I was scammed! The box is somwhat lightweight so I thought nothing was in it. even took a video for proof. Haha! But, lo and behold! The item is so beautiful! I delayed the review for a day to test the item and, I was not disappointed! The sound was quite good and crisp considering it's price. I haven't tried it outside yet but, I hope it delivers. Overall, delivery fast. Item, superb (as of now, hopefully, in the long run too!) Will definitely order again next time!",
    "The case hinge easily broke so i had to put a tape to serve as a hinge. Sound quality is muffled. The treble is dead. The connection easily interrupted and disconnects / reconnects shortly even when your phone is near you.  For its price, its definitely cheap and so is the quality. The audio prompt is in Chinese.",
    "JUNK!!!!! DON'T BUY THIS!!!!!!!  I JUST HAD IT YESTERDAY AND IT'S ALREADY SKIPPING SOUNDS. WASTE OF MY MONEY"
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
    aspects = set()
    p_stemmer = PorterStemmer()

    for sentence in sentences:
        review_text = BeautifulSoup(sentence, features="html.parser").get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)
        words = letters_only.lower().split()
        stops = set(stopwords.words('english'))
        stops.update(['app', 'shopee', 'shoppee', 'item', 'items', 'seller', 'sellers', 'bad', 'thank', 'thanks', 'delivery', 'package'])
        meaningful_words = [w for w in words if w not in stops]
        # meaningful_words = [p_stemmer.stem(w) for w in meaningful_words]
        final_text = " ".join(meaningful_words)
        doc = nlp(final_text)
        for sent in doc.sents:
            target = []
            for token in sent:
                if (token.dep_ == 'nsubj' or token.dep_ == 'dobj') and token.pos_ == 'NOUN' and token.ent_type_ == '':
                    target.append(token.text)
            if target:
                aspects.update(target)
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
        stops.update(['app','shopee','shoppee','item','items','seller','sellers','bad', 'thank', 'thanks', 'delivery', 'package'])

        # Remove stopwords
        meaningful_words = [w for w in words if w not in stops]

        # Stem words
        meaningful_words = [p_stemmer.stem(w) for w in meaningful_words]

        # Join words back into one string, with a space in between each word
        final_text = pd.Series(" ".join(meaningful_words))

        # Generate predictions
        pred = model.predict(final_text)[0]
        proba = model.predict_proba([pd.Series.to_string(final_text)])[0]

        positive_prob = proba[0]
        negative_prob = proba[1]

        overall_prob = positive_prob - negative_prob

        if pred == 1:
            output = "Negative"
        else:
            output = "Positive"
    
        sentence_sentiments.append([text, output, overall_prob])

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
    word_model = KeyedVectors.load_word2vec_format("C:\\Users\\kreyg\\OneDrive\\Documents\\word2vec-model\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin", binary=True, limit=1000000)
    #word_model = KeyedVectors.load_word2vec_format("Aspect-Extraction/GoogleNews-vectors-negative300.bin", binary=True, limit=500000)

    # Convert aspects to word vectors
    aspect_vectors = []
    for aspect in aspect_list:
        if aspect in word_model:
            aspect_vectors.append(word_model[aspect])
        else:
            print(f"Warning: Aspect '{aspect}' not in vocabulary.")

    # Cluster word vectors using k-means
    kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
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
        aspect_scores = []
        for aspect, sentiment in sentenceMaps.items():
            if aspect in aspects:
                overall_probs = [sentence[0][2] for sentence in sentiment]
                aspect_scores.extend(overall_probs)
        if aspect_scores:
            avg_score = sum(aspect_scores) / len(aspect_scores)
            sentiment_label = interpretSentimentScore(avg_score)
            mapScores[aspect_label] = {
                # 'sentiment_scores': aspect_scores,
                'average_score': avg_score,
                'sentiment_label': sentiment_label
            }
    return mapScores

def interpretSentimentScore(score):
    if score > 0.2:
        return 'Positive'
    elif score < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

def countSentiments(sentence_maps, grouped_aspects):
    count_dict = {}
    for aspect_label, aspects in grouped_aspects.items():
        count_dict[aspect_label] = {}
        for aspect, sentiment_list in sentence_maps.items():
            if aspect in aspects:
                pos_count = 0
                neg_count = 0
                count_dict[aspect_label] = {}
                for sentence in sentiment_list:
                    if sentence[0][1] == 'Positive':
                        pos_count += 1
                    else:
                        neg_count += 1
                count_dict[aspect_label] = {'pos-count': pos_count, 'neg-count': neg_count}
    return count_dict
# def getOverallSentiment(result):
#     new_dict = {}
#     for aspect_label, sentiment_dict in result.items():
#         positive_sentiment = 0
#         negative_sentiment = 0
#         for aspect, sentiment in sentiment_dict.items():
#             if 'positive' in sentiment.lower():
#                 positive_sentiment += 1
#             elif 'negative' in sentiment.lower():
#                 negative_sentiment += 1
#         total_sentiment = positive_sentiment + negative_sentiment
#         if total_sentiment > 0:
#             overall_sentiment = positive_sentiment / total_sentiment
#         else:
#             overall_sentiment = 0
#         new_dict[aspect_label] = {'pos-count': positive_sentiment, 'neg-count': negative_sentiment, 'overall-sentiment': overall_sentiment}
#     return new_dict

def listSentences(sentence_maps, grouped_aspects):
    text_list = {}
    for aspect_label, aspects in grouped_aspects.items():
        text_list[aspect_label] = {'Positive': [], 'Negative': []}
        for aspect, sentiment_list in sentence_maps.items():
            if aspect in aspects:
                for sentiment in sentiment_list:
                    sentence = sentiment[0][0]
                    sentiment_label = sentiment[0][1]
                    if sentiment_label == 'Positive' and sentence not in text_list[aspect_label]['Positive']:
                        text_list[aspect_label]['Positive'].append(sentence)
                    elif sentiment_label == 'Negative' and sentence not in text_list[aspect_label]['Negative']:
                        text_list[aspect_label]['Negative'].append(sentence)
    return text_list

        

def getABSA(sentences):
    my_aspects = getAspects(sentences)
    my_groupedAspects = groupAspects(my_aspects, sentences)
    my_dict = createAspectSentimentDict(my_groupedAspects, mapSentences(sentences))
    return my_dict

def getListSentences(sentences):
    my_aspects = getAspects(sentences)
    my_groupedAspects = groupAspects(my_aspects, sentences)
    # my_dict = createAspectSentimentDict(my_groupedAspects, mapSentences(sentences))
    return listSentences(mapSentences(sentences), my_groupedAspects)

def getCountSentiments(sentences): 
    my_aspects = getAspects(sentences)
    my_groupedAspects = groupAspects(my_aspects, sentences)
    # my_dict = createAspectSentimentDict(my_groupedAspects, mapSentences(sentences))
    return countSentiments(mapSentences(sentences), my_groupedAspects)

absa = getABSA(sentences)
list_sen = getListSentences(sentences)
count_sen = getCountSentiments(sentences)

# for aspect_label, nested_dict in absa.items():
#     print(aspect_label + ":")
#     for aspect, sentiment in nested_dict.items():
#         print("  {} -> \n   {}".format(aspect, sentiment))

print(json.dumps(absa, indent=1))
print(json.dumps(list_sen, indent=1))
print(json.dumps(count_sen, indent=1))


# print(json.dumps(getSentiment(sentences), indent=1))
# print(getABSA(sentences))
# print()
# print(json.dumps(listSentences(sentences), indent=1))
# print(getAspects(sentences))
# my_aspects = getAspects(sentences)
# my_groupedAspects = groupAspects(my_aspects, sentences)
# group = mapSentences(sentences)
# print(json.dumps(group, indent=1))
# my_dict = createAspectSentimentDict(groupAspects(my_aspects,sentences), mapSentences(sentences))
# for aspect_label, nested_dict in my_dict.items():
#     print(aspect_label + ":")
#     for aspect, sentiment in nested_dict.items():
#         print("  {} -> \n   {}".format(aspect, sentiment))
#     print()

#print()
# print(json.dumps(getOverallSentiment(my_dict), indent=1))



