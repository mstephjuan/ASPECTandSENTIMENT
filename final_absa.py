import spacy
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import pickle
from collections import Counter
import json
import re

nlp = spacy.load("en_core_web_lg")
model = pickle.load(open("SentimentModel/modelCraig.pkl", 'rb'))

stops = set(stopwords.words("english"))
stops.update(['app', 'shopee', 'shoppee', 'item', 'items', 'seller', 'sellers', 'bad', 'thank', 'thanks', 'delivery', 'package', 'things', 'family', 'damage',
                'niece', 'nephew', 'love', 'error', 'packaging', 'way', 'wife', 'husband', 'stuff', 'people', 'know', 'why', 'think', 'thing', 'kind', 'lots',
                'pictures', 'picture', 'guess', 'ones', 'tweaks', 'joke', 'specs', 'work', 'play', 'macbook', 'bit', 'modes', 'mode', 'time', 'times', 'day', 'problem', 
                'want', 'others', 'issue', 'see', 'reason', 'reasons', 'lot', 'lots', 'others', 'issues', 'issues', 'problem', 'problems', 'way', 'ways', 'day', 'days', 'tis', 'puts', 'user', 'hassle', 'gtav', 'means', 'lengths', 'world', 'skim', 'person', 'computer', 'screws', 'years', 'game', 'games', 'lap'])

# List of Reviews
reviews = [
    "I purchased this laptop with the primary intent to be able to run ML models and use it for the DS program that I'm in. I wanted something upgradeable with solid core hardware and this fit my budget. My comments about the machine are not from a gaming perspective.\n\nScreen\nI saw others comment that this wasn't the best screen for gaming. It's not the brightest thing in the world and almost reminds me of a matte screen - even though it is not.\n\nKeyboard\nI've typed on worse (I'm looking at you, Macbook) and better (my trusty Lenovo) and this keyboard is okay. If I had to give it a grade it would be mid-point. I'm thinking of key travel and comfort - it kind of falls in the middle. I sometimes lose my home keys, so it can take a moment to reorient. I do like the number pad, it comes in handy here and there. Using mostly a traditional keyboard I hit Home and End quite a bit. This keyboard has them paired up, so you have to hit Fn if you want to End. I work a lot in code, data, and spreadsheets, and it can be frustrating at times. Keep in mind my use case is probably not the average user and it's not THAT bad.\n\nThis sucker is a fingerprint magnet, I have not spent any significant time to find the proper cleaning technique. :P\n\nUpgradeability\nAs I mentioned, I wanted something I could upgrade, which I did. From some of the other reviews it seems like these shipped with a single 16GB stick of RAM. My unit came with two 8GB sticks, so the extra 16GB stick that I ordered with the unit was returned. I ended up buying a 64GB kit (CT2K32G48C40S5) that works just fine. I am unable to see what timings are being used - cannot see in BIOS, nor via CPUID. I thought that was odd, but everything I'm seeing shows that I have 64GB. For most people this is way overkill, but for my needs I can end up utilizing quite a bit.\n\nThe SSD that came with the unit was a 512GB - I cloned it over to 1TB Samsung 980 and threw in another 1TB Samsung 980.\n\nI should mention that the back was extremely easy to remove and upgrading the components was super easy. It should be noted that one of the screws does not come out of the back all the way - by design. There are also multiple screw lengths, so be very aware of what goes where.\n\nOS\nIt comes with Win11 - frankly not a fan of it. I have considered wiping and trying to put Win10 or Linux on, or even dual boot. For now I'll leave it, but wanted to make sure you know it's coming with Win11.\n\nSoftware\nHas a bit of bloatware, but it's minimal. I was expecting there to be more, but it's not too bad.\n\nBattery life\nI think if you try to game on battery you'll be disappointed. Firing up local databases and running models gives me 3~ish hours of battery. If I push it hard it will probably drop. Now, I leave it in \"performance mode\" most of the time, so I'm sure that has an effect on it. If you're a gamer I hope you're looking at other reviews :D\n\nOverall thoughts\nI like this laptop, it's very functional and gave me decent bang for the buck performance. Even spending a bit extra on upgrades left me in a good spot. If you're shopping for this one, good luck on your decision!",
    "It's pretty good, especially if you can get it for sub $1,150. Has some annoying caveats to it's otherwise great specsheet.\n\nPositives:\n\n* Has a great processor, overkill for most things. The 3060 will bottleneck this way before the CPU does.\n* Has a dedicated GPU\n* It doesn't look as gamery in person as it does in the product pics, which is good.\n* It can charge off USB C and features a decent selection of IO, media keys, etc.\n* It isn't too heavy, and is pretty thin all things considered.\n* Keyboard is decent, nice inclusion of numpad\n\nNegatives:\n\n* NO WINDOWS HELLO CAMERA OR FINGERPRINT SENSOR.\n\n* The battery life isn't great, possibly due to bloatware. (about 2.5 - 4 hours in power saving mode with keyboard LEDS off)\n* There is a lot of bloatware running in the background. Have fun cleaning this up so it doesn't eat your battery.\n* It doesn't come with a USB C charger, so if you don't wanna lug the battery brick around, you are gonna have to buy a different charger.\n* This laptop doesn't support S3 sleep, it uses windows modern sleep. This drains the battery 7%/night. Also, it has useless LEDS that show the SSD is still running in sleep mode.\n* The screen and speakers aren't the loudest or brightest.\n* Doesn't have a 120hz screen refresh rate, ONLY 60 or 144, which it can't hit in many games.\n\nOverall, it's a great laptop, though the inclusion of Amory crate, Mcafee, and other bloat makes it a hassle to clean up. The batterylife isn't great because it doesn't support s3 sleep and W11 has problems with performance profiles. This means when you unplug it, it won't automatically go into 60 hz mode, disable most P cores, turn battery saver on, turn off all silly gamer LEDS. Which means you are really eating into that battery. For the price and weight, this laptop isn't bad. The fact it can charge off 100W type C is really nice. The specs on paper are great, and it doesn't seem to throttle or run into performance issues.\n\nIdeal specs for this laptop would a Intel i7 1260P, 100wh battery, 3060, cold air intake on the top of the laptop, USB C charger included, since it can do up to 240W over C. Slightly move IO around so it wasn't all on one side. No bloatware and all armorycrate functions moved to BIOS/UEFI. Support for S3 sleep so that it actually goes to sleep instead of just still running, lol.",
    "I've wanted something that I could use as an everyday computer for work and use for gaming in my off time, something that could combine productivity and gaming in a compact design. I did a lot of research and found several (Razor Blade, Sager) but they were just too far out of my price range. I was worried I was being unrealistic for my budget. Then I came across the 2022 ASUS TUF Dash 15 (FX517ZM-AS73).\n\nFrom a physical standpoint I find the form factor is great in the hands and taking it mobile is a breeze thanks to it's compact, lightweight and super sleek design. I can easily pop it into my messenger bag when I need it on the go and when at home I can hook it up to a secondary display and enjoy gaming on it like I would a desktop.\n\nUnder the hood it's clear ASUS wanted to make an impression. The hardware is impressive, yes, but how it synergies together is the most impressive part. Across the board performance is near instantaneous and (if power options and Armory Crate are utilized well) the battery life is also quite impressive when used for productivity tasks, too.\n\nThe hardware is an ensemble cast with an Intel Core i7-12650H, 16GB of lightning fast DDR5 memory, a 512GB M.2 SSD and a 6GB GeForce RTX 3060 Mobile to round it out. It should be mentioned that while the computer itself supports Gen4 M.2 drives the stock 512GB SSD is Gen3 - likely to cut costs - however don't let that deter you as for an O/S drive it works great. Also, in a nice change of pace this laptop can be completely upgraded and you won't find any hardware soldered to the board. I have since added a 1TB Samsung 990 Pro in the second M.2 slot (for game installs, storage) and could not be happier with the near-instant performance.\n\nSpeaking of performance I have yet to find a game title that I am unable to max out the settings on, even when hooked up running UHD resolutions on my secondary display (I have an LG UltraGear 27GL83A-B) which takes advantage of the G-Sync support for silky smooth response times. I've not yet even needed to use the \"Turbo\" mode as everything gets handled quite comfortably using the \"Performance\" setting.\n\nASUS did their homework when they designed the 2022 iteration and you can tell they were paying attention, the 2022 Dash 15 corrects many of the complaints from the previous years model and then some. The new cooling system design is incredibly effective and helps cut back on a lot of fan noise, even when under load (using the \"Performance\" setting) the fans when kicked up are barely audible while keeping everything nice and cool.\n\nI'm very impressed with this machine and can easily recommend it to anybody interested in this model. No matter if you're looking for a dedicated gaming laptop or something to use for daily productivity tasks and some gaming on the side the 2022 ASUS TUF Dash 15 (FX517ZM-AS73) is hard to beat for performance and value.",
    "I bought this laptop on the 3rd of April and got it on the 11th even though tis only been 2 days i can tell a few things, any expectations you have of this laptop regarding gaming will be met and some more This beauty will give you frames like you havent seen before( i came from a gtx 1060 and i7 8th gen) and the battery life not so sure but when connected to an external display will be 2 hours and 40 minutes but on its own it will probably be double that for a gaming rig this will rival quite a few desktops highly recommend now the flaws when on turbos this sounds like a jet (i mean it) and 1 more flaw i have a 165 hz external monitor and when connected to it the laptop couldnt only support 120 hz i got it checked and the fault was in the laptop nonetheless this is still a great buy\nI tested 2 games and hear are the fps\nGTAV - VeryHigh/Ultra settings =120+fps ( i cant say since my external monitor is capped but it can easily go more than 120 fps)\nValorant- Shooting Range=600+fps Ingame=(Low settings)=400+/)High settings)=300+",
    "I got this laptop mainly for one game. It is very lightweight, unexpectedly light, and beautiful. Sleek, the keyboard blinks with its light, and the track pad and keys are smooth.\n\nIf I unplug it, my laptop cannot stay on for over an hour when playing my game. It drops so quickly. Even with battery saver, brightness low, and every other setting to accommodate doesn't help. I played for an hour unplugged and by 20 mins, it dropped 30%, and by the hour it wad about 16%% left.\n\nIt gets pretty hot. If left on my lap, I had to keep moving it cause my skim felt burning. The fans are also kind of loud. Not very loud but very noticeable.\n\nThe track pad is nice and big but with that, my hand kept touching it when I'd use my fingers and it didn't quite register I was moving the mouse. I've had to get an actual mouse to use cause my hand kept touching it when I tried not to.\n\nOverall, I still love the laptop, but it is not as portable as I wanted or expected. I can only really play games if it's plugged in, or if I have 30 minutes.",
    "Really nice laptop, so far. I purchased this in August and put off writing a review until after I'd used it for several months. I wanted a laptop with a dedicated GPU, a fast 12th Gen Intel CPU, and Windows 11, but wanted to keep the price under control. This laptop fit the bill perfectly.\n\nMy previous laptop was a 2-in-1 with a slightly larger touch screen. While I occasionally miss that capability, the higher resolution screen and the addition of the dedicated NVIDIA GPU make up for it. The screen is great - the high resolution is very sharp, and the screen is nice and bright with good colors. This laptop is very fast and more than adequate for the games that I play, or any other use I have for it.\n\nIf battery life is your primary concern, you will be happier with a lower powered laptop, but you probably wouldn't be shopping for a gaming laptop. The battery life is not great but it's acceptable for a gaming laptop, if you set up the Power and GPU settings correctly.\n\nWhen using the NVIDIA GPU, this laptop is very power hungry, but that's to be expected. The Windows 11 \"Display Graphics\" settings can be used to set applications to specifically run on the Intel GPU (\"Power Saving\") or specifically run on the NVIDIA GPU (\"High Performance\"). The Intel GPU is more than adequate for daily use (and some lower resolution gaming), and uses significantly less power.\n\nAdd a dock and external monitors, and this laptop is also a more than adequate workstation replacement.\n\nMy only personal quibble with this laptop is the amount of preinstalled software. Asus preinstalls several applications to customize the user experience (and provide data to Asus) -- not all of these are useful, and all of these are designed to require user identifiable connections to Asus, and agreeing to their terms of service. These aren't terribly hard to uninstall (so they will no longer pester you to register) but it's always irritating.",
    "So far I like this computer a lot, I think that performance is good. The cost for that performance however, is the battery, it will not last more than a couple of hours unless you put battery saver. The fast charging feature is very good to compensate, but this only work if you can find where to connect. Overall I really like this machine, perform very good for my work and when i need to take a break I play some games without issues. I do recommend this product.",
    "Rese\u00f1a con una semana de uso, actualizare si veo mas fallas:\n\nEl equipo funciona bien, y tiene los componentes descritos, aun que en ocasiones he sentido tirones. Siento que pude haber conseguido algo similar por menos dinero.\n\nRaz\u00f3n por la que le quito 1 estrellas\n- Un detalle importante, es que los botones est\u00e1n con la distribuci\u00f3n del sistema ingles. (por eso le doy 4 estrellas, siento que deber\u00edan incluir la opci\u00f3n de elegir que distribuci\u00f3n se desea).\n-A veces siento tirones(aun que son m\u00ednimos) y he tenido problemas de conexi\u00f3n con YouTube(un par de veces), me ha pasado con simplemente usar google chrome y microsoft edge, puede que sea por actualizaciones necesarias o problemas de servidor o conexi\u00f3n. Espero que ande bien en otras aplicaciones.\n\nOtras apreciaciones:\n-El conector del cable de alimentaci\u00f3n el\u00e9ctrica esta en la mitad del equipo, estaba acostumbrado a laptops con el conector en una esquina.\n-El teclado es completo, tiene el teclado num\u00e9rico, y la pantalla es mas alta de lo que estaba acostumbrado.\n-Dise\u00f1o elegante, silencioso de momento.\n-No conoc\u00eda el puerto thunderbolt4, creo que no lo usare, y siento que eleva el precio del articulo.",
    "The media could not be loaded.\nWe updated the driver for the graphics card to see if that would rid the display of the weird strip of light on the left side of display. It didn't repair the issue. We tried calling Asus but they seem to only have a 'chat' box for communicating and after waiting for an hour we gave up as no one from ASUS responded. Obviously, this flashing strip of light at the edge of the display screen is mentally fatiguing and unhealthy. Will see if ASUS replies this weekend or if we need to return the laptop.\n\nUpdate: We called ASUS Tech Support and they solved the problem by resetting the bios. All good now.",
    "Love this computer! Don't use it for gaming but, it's so quick, light for traveling vs previous laptop, great speakers, great for lots of photos and works well with Zoom meetings.",
    "Mi hermana la usa para sus dise\u00f1os y dibujos gr\u00e1ficos no ha tenido problemas, va super bien",
    "This is a great discrete gaming laptop for school, which is the main reason I purchased it. It gives plenty of options for battery saving and options for how loud the fans are, as well as the typical Nvidia options for your GPU. I did have to upgrade the storage, but is not difficult by any means, just required me to remove a couple of screws and put in the new SSD and put the screws back in. However there are a few downsides to this computer. If you plan on using it on any sort of incline, good luck. The screen doesn't fold back very far which I have found a little annoying at times. Another small complaint is the type of plastic the computer and keyboard are made of. It feels pretty sturdy, however it gets dirty from everything. I consider myself a relatively clean person and I like my hands to be clean, but I find myself having to wipe this keyboard off weekly. No matter how clean your hands are this keyboard manages to get oily and dirty. Other than those small details this is an amazing mid-range laptop for gaming, travel, and school. It is also easily upgradeable if the specs (Ram and SSD Storage) are not to your liking.",
    "Great gaming cpu!!",
    "The laptop exceeded all my expectations. It is a super-fast machine with an extremely well heat dissipation system.",
    "For gaming the results are beyond than expected, run new games smoothly and the fans work really well keeping the laptop fresh, also I use this laptop for coding and it works awesome everything is super fast, really recommended.",
    "Bought this as a mobile rig, and it pretty much ticks all the boxes. only issue, and im not sure if its a user error on my end or just the overall hardware configuration, i burn through battery life like its running on AAs. this is when im trying to do non gaming tasks. notebook is very sleek, the weight is not bad feels balanced. the hardware is more than sufficient. this PC has the essentials and none of the luxuries. perfect for me to throw in a back pack. i was also impressed to see ASUS offers a year warranty that covers raw damage. this pc puts in work and pumps out heat when in turbo mode (plugged in) but overall not to bad when doing light gaming. handles CoD MW2 excellent, and even FiveM (GTA). after shopping around and about to purchase a 2000$+ unit, im happy i found this for half the cost.",
    "Love everything...nothing to hate",
    "Bought to replace my Asus N55S.\nFirst Gen I7 processor with 8 gigs of ram.\nAfter almost 10 years I wanted something better. Wasn\u2019t going to buy a Dell I5 or something with 4,6,8 gigs of ram. Dell is betting you will buy a new computer in a few years. Bought this one on Amazon Warehouse saved about $300.",
    "I've been a long-time ASUS fan. I've got a self-built desktop that uses numerous ASUS components, and have used their laptops in the past (my family business also bought a couple Vivobooks). The designs are usually top notch & well built. This one has a solid LCD housing without too-much flex. The battery life is reasonable for this system; the \"silent\" mode keeps things quiet & cool for non-gaming and non-video/audio production tasks. \"Performance\" mode (predictably) runs the battery down quickly but this mode is best used when connected to a power outlet. The fans are a little loud under heavy loads, but that's also to be expected in such a slim chassis. The speakers are competent but not stellar. The display has a great refresh rate & good viewing angles. I think the i7-12650H model was the right choice as it still gives you all 6 Performance Cores compared to the 12700H, you're losing 4 E-cores but as a mostly stationary/occasionally portable gaming system it's not really a loss. The TB4 & multiple USB ports (that don't block each other by being too close) are all USB 3 & up, no USB 2 here. My model has an SK Hynix Gold P31 SSD (512GB) & I'm plenty happy with that performance. I'll but putting in a second NVMe drive later since the slot on the board is open. Keyboard is good, the touchpad has an excellent feel & spacious.\n\nI've used it with Windows 11 and Manjaro 21; it performance at the expected high level in both OSes.\n\nNo complaints, definitely worth it. It's a great system and I think you'll enjoy using it very much. If you can find it on a sale for $1100 (USD) then it's a definite buy over other brands' 15-inch models.\n\n*Note that a 90-day warranty extension is offered by ASUS for this review.",
    "The only issue is the battery \ud83d\udd0b it don't give me a lot Like 4 to 5 hours otherwise is good price, good laptop \ud83d\udc4d"
]

preprocessed_reviews = []

def Preprocessing(reviews):
    for review in reviews:
        review = review.lower()
        review = re.sub(r'\d+', '', review) # remove numbers
        review = re.sub(r'[^\w\s]', '', review) # remove non-word characters
        review = re.sub(r'\s+', ' ', review) # remove extra whitespaces
        review = review.strip() # remove leading and trailing whitespaces
        meaningful_words = [w for w in review.split() if not w in stops] # remove stopwords
        final_text = ' '.join(meaningful_words) # convert list of words to string
        preprocessed_reviews.append(final_text)
    # print(json.dumps(preprocessed_reviews, indent=2))
    return preprocessed_reviews

def ExtractAspects(reviews):
    new_reviews = Preprocessing(reviews)
    doc = nlp(' '.join(new_reviews)) # convert list of reviews to string (nlp only accepts string)
    aspects = set() # set() to remove duplicates
    lemmatizer = WordNetLemmatizer()
    for token in doc:
        if token.pos_ == 'NOUN' and token.dep_ == 'nsubj':
            if token.text not in stops: # does not accept aspect words that are stopwords
                # check if the noun is the head of the subject subtree
                if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                    subject_words = [t for t in token.subtree if t.dep_ == 'nsubj']
                    subject_words.sort(key=lambda t: t.i)
                    if subject_words[0] == token:
                        aspect = lemmatizer.lemmatize(token.text, pos='n')
                        aspects.add(aspect)
                else:
                    aspect = lemmatizer.lemmatize(token.text, pos='n')
                    aspects.add(aspect)
            else:
                stops.add(token.text)
    # print(json.dumps(list(aspects), indent=2))
    return list(aspects)

def ExtractTopAspects(reviews, aspects):
    aspect_counts = Counter()
    for aspect in aspects:
        count = 0
        for review in reviews:
            if aspect in review:
                count += 1
        aspect_counts[aspect] = count
    top_aspects = [aspect for aspect, count in aspect_counts.most_common(6)]
    print(json.dumps(top_aspects, indent=2))
    return top_aspects

def ExtractAspectPhrases(reviews, top_aspects):
    new_reviews = []
    for review in reviews:
        # processed_review = re.sub(r'[^\w\s\.]', ' ', review)
        # processed_review = re.sub(r'\s+', ' ', processed_review)
        # processed_review = processed_review.strip()
        # # Split the review into sentences
        sentences = sent_tokenize(review)
        # print(json.dumps(sentences, indent=2))
        new_reviews.extend(sentences)
    aspect_sents = {}
    for aspect in top_aspects:
        aspect_sents[aspect] = []
        for review in new_reviews:
            if aspect in review:
                aspect_sents[aspect].append(review)
    print(json.dumps(aspect_sents, indent=2))
    return aspect_sents

def SentimentAnalysis(phrases):
    aspect_sentiments = {}
    for aspect, sentences in phrases.items():
        aspect_sentiments[aspect] = []
        for sentence in sentences:
            new_sentence = Preprocessing(sentence)
            final_sentence = pd.Series(" ".join(new_sentence))
            score = model.predict(new_sentence)[0]
            proba_score = model.predict_proba([pd.Series.to_string(final_sentence)])[0]
            if score == 1:
                output = 'Negative'
            else:
                output = 'Positive'
            aspect_sentiments[aspect].append((sentence, output))
    print(json.dumps(aspect_sentiments, indent=2))
    return aspect_sentiments
aspects = ExtractAspects(reviews)
top_aspects = ExtractTopAspects(reviews, aspects)
aspect_phrases = ExtractAspectPhrases(reviews, top_aspects)
aspect_sentiments = SentimentAnalysis(aspect_phrases)