from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import spacy
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from collections import Counter
import json
import re
# from concurrent.futures import ProcessPoolExecutor
import torch
import random
# from transformers import pipeline
# generator = pipeline('text-generation', model='gpt2')
# summarizer_model = pipeline('summarization', model='facebook/bart-large-cnn')

from transformers import LEDForConditionalGeneration, LEDTokenizerFast
tokenizer = LEDTokenizerFast.from_pretrained('pszemraj/led-base-book-summary')
summarizer_model = LEDForConditionalGeneration.from_pretrained('pszemraj/led-base-book-summary')

# max_length = 512

nlp = spacy.load("en_core_web_lg")

stops = set(stopwords.words("english"))
stops.update(['app', 'shopee', 'shoppee', 'item', 'items', 'seller', 'sellers', 'bad', 'thank', 'thanks', 'delivery', 'package', 'things', 'family', 'damage',
                'niece', 'nephew', 'love', 'error', 'packaging', 'way', 'wife', 'husband', 'stuff', 'people', 'know', 'why', 'think', 'thing', 'kind', 'lots',
                'pictures', 'picture', 'guess', 'ones', 'tweaks', 'joke', 'specs', 'work', 'play', 'macbook', 'bit', 'modes', 'mode', 'time', 'times', 'day', 'problem', 
                'want', 'others', 'issue', 'see', 'reason', 'reasons', 'lot', 'lots', 'others', 'issues', 'issues', 'problem', 'problems', 'way', 'ways', 'day', 'days', 'tis', 'puts', 'user', 'hassle', 'gtav', 'means', 'lengths', 'world', 'skim', 'person', 'computer', 'screws', 'years', 'game', 'games', 'lap', 'ect', 'con', 'cons', 'camera'])

# List of Reviews
reviews = [
    "I am a DSLR photographer. I do a lot of bird photography. I have a nikon 200-500mm ED VR with a Nikon D750, a Canon 6D Mark II with a Sigma 150-650mm, a Pentax K-1 with a Pentax 150-450mm AW, and a Canon 77D with a Canon 400mm L, a Sigma 100-400mm with a Nikon D3400.\n\nI am describing my gear to the reader so he/she will see how my review will be influenced by my current setup. Anytime I carry any of my gear mentioned above, I am looking at a good 4 to 8 lbs of gear strapped on my chest and hiking miles of trail. The Nikon Coolpix P1000 can never beat the image quality of a proper DSLR. The Nikon Coolpix P1000 is NO MATCH against sharpness and clarity of the images that my DSLRs produce. NO MATCH...and that is understandable. BUT...sometimes, I feel like I don't want to haul heavy gear. Sometimes, I just want to enjoy the scenery and the walk but still having the capability to take photos. Sometimes, I just want to be silly and just take photos of anything...a rock, a bird, a raccoon paw print, the bark of a tree. Enter the Nikon P1000.\n\nThe Nikon Coolpix P1000 is an excellent point and shoot camera as long you know its limitations. The camera uses a tiny sensor, a smartphone sensor, so low light is the first challenge already. Now, there is a way to squeeze out light from this sensor by going Manual, or Aperture Priority and adjusting the ISO to 3200 or 6400 (max ISO). The next challenge is the 125X zoom power. You can get great quality images between 24-1500mm handheld as long as you keep your shutter speed around 500-1000, and then push the ISO a bit. Shooting handheld with this camera is doable as long you keep your zoom range within a reasonable range. Once you push pass 1500, you are now faced with hand shaking, wind, your breathing, the twitch of your arm, they start multiplying ten-fold as you push your zoom range closer to 3000mm. The next question is: do you need a tripod to push 3000? This is up to you. Remember though, when you push 3000mm even with a tripod, the quality of the image degrades because of the distance and size of the subject. This is a fact and it must be accepted by the user because it's just how the technology was built into this system, and the reality that 3000mm will start picking up heat shimmers, wind etc start showing up. It is the nature of super-telephoto lenses- any lens in the market. Do not be upset or disappointed when you see these issues at higher zooms, it is not Coolpix P1000s fault. It's just the nature of super zooms.\n\nLet's get the weight issue out of the way. It's about 2lbs maybe? Not sure because I am so used to carrying heavy camera gear that having this Coolpix P1000, to me, is very light. It might feel heavy for others not used to carrying heavy camera gear. But for me, this camera is light as \"feather\" so to speak, in relation to my heavy gear. Anyway, the point is..it's not going to break your hip or back. After you use this camera on several trips and hikes, your body will eventually adjust to the weight.\n\nWhat about images within ranges of 24-1500? I can say they look good to great. I know this is a point and shoot camera but if you really want to get the best out of this camera, the user needs to learn other forms of shooting styles, particularly using the different shooting modes- M, A, S. Trust me. Learning these other shooting modes will really bring out Nikon Coolpix P1000s capabilities from a point and shoot to a intermediate \"DSLR\"-like features.\n\nThe Nikon Coolpix P1000 also has features like RAW format. Note that if you are using RAW format, the digital zoom is not available. Optical zoom at RAW format only goes up to 3000mm. Note that there are two zooms...Optical and Digital. User needs to read the full manual just in case they notice a function doesn't work; chances are the functionality isn't available at that moment because the camera locked you out because of a setting you made. Again, these are all described in full manual. The Quick manual that came with the camera is just a quick reference guide. You will need to download the full manual free from Nikon USA's website. This full manual will tell you EVERYTHING that you need to know about the camera.\n\nThe Coolpix also has other \"DSLR\" like features like Spot metering, Center-weighted, Matrix. It also has manual AF select, Spot focusing; typical focusing features found in proper DSLRs. You can also set the ISO number. If you want to change ISO without going thru the MENU interface, you can assign the FN button to pull up the ISO numbers. Lots of neat stuff. All of thse are described in the full manual.\n\nWhat about battery use? You will need an extra battery. Please please use Nikon batteries. Do not use third-party batteries. Since you've already invested $1000 on a camera, there's no reason for you to go cheap on the battery. Bite the bullet and get a Nikon battery. Don't get those Wasabi or Watson batteries. Spend a bit more on real genuine Nikon batteries. Rated at 250 shots according to Nikon. Some ways to save battery power is to not use the Monitor all the time. Have the auto power off setting at 30 seconds.\n\nThe Nikon Coolpix comes with a strap, a Nikon battery, a USB cable, a USB charging adapter, warranty card, manual, lens hood, lens cap.\n\nAccessories you need to start with? I strongly suggest getting a UV lens filter for this camera. The filter size is 77mm. I also suggest using a shoulder strap instead of the neck strap that came with the Nikon Coolpix P1000. Never ever strap a camera around your neck. I really think neck straps should not be used for cameras of this size as this causes neck strain. Please get a good shoulder strap. You might want to get a wired Nikon MC-DC2 remote release cord. It's no expensive. Do not get the Nikon Bluetooth remote cord- too expensive. Start with the Nikon MC-DC2. This remote cord helps you press the shutter button remotely (it's wired) to reduce vibration/shaking when you are focusing at a distant object; for example, the moon. You might want to invest on tripod. Just make sure the camera is well-tethered to you when attaching this camera to a tripod.\n\nOverall, I am happy with my Nikon Coolpix P1000. I am still testing all its capabilities. I recommend this camera to those looking for an all in one camera. Again, I can't stress enough about knowing its limitations. It takes great images as long as you know how to utilize its \"powers\".\n\nAnd finally, practice practice practice. The camera is just a device, majority of the work will have to come from the user for this camera to come up with great images. Highly recommended camera.",
    "Wildlife around my house. Everything I like.",
    "Update: After several months and over 10,000 photos of birds and other wildlife, I have upgraded my review to 5 stars. This is clearly the best birding camera I have ever owned, including a Canon 7D DSLR camera with a 400mm zoom lens. I have found that if I turn the display inward (facing into the back of the camera) it will stay off. The viewfinder also stays off unless I lift the camera to my face to view it. So, I can leave the camera turned on while I take nature walks without running down the battery as fast. This allows me to capture pictures much faster before a bird or other subject moves out of sight. The battery still runs down after about 1.5 hours, but I keep a spare battery in my pocket. This allows me to catch birds in flight almost as easily as the 7D. For birds sitting still, especially at long distances, the P1000 is way better, hands down. Low light performance is also way better with the P1000. Videos of an eagle nest at 1/3 mile using a tripod are very good, as good as our $1600 spotting scope. Viewing the movie as it is being taken on the display screen is also about as good as our spotting scope.\n\nOriginal review:\nThe best part about this camera is the electronic viewfinder. It is bright and detailed enough to make the camera like a hand-held image-stabilized spotting scope with automatic focus and brightness adjustment. I watched an eagle pair working on their nest from 1/3 mile away. The view through the view finder was almost as good as a $1600 spotting scope (85mm x 60). That alone is worth the price. I tried to upload a hand-held movie of it, but the upload conversion turned it into garbage. The original movie was almost as good as the view through the viewfinder.\n\nThe still pictures at long zoom look substantially less detailed than what the eye sees through the viewfinder. That tells me that the image processing (frame averaging, etc.) in human vision is better than in the camera. So, we can expect that future super-zoom cameras will improve dramatically as the image processing becomes faster and more sophisticated.\n\nLow light performance, especially autofocus at long zoom settings is bad, but it is easy to quickly switch to manual focus. The focus ring is easy to operate.\n\nI compared this to my old Nikon P900 and found no obvious improvement in zoom reach. The picture is bigger at max zoom, but the amount of detail is about the same. So the increase is size, weight was for naught.\n\nI still consider this a substantial upgrade because of the brighter and more detailed viewfinder and the easy manual focus. That makes a huge difference in finding the bird or other target, composing the picture, and adjusting brightness and focus.\n\nIf I could redesign the P900 to my liking, I would make its grip bigger to fit my large hands, make the battery bigger, make the the battery charge indication more accurate, replace the viewfinder with the one from the P1000, and give it an adjustment ring and manual focus like the P1000.",
    "So this is a quick overview of how this camera will compare against the P900 for WILDLIFE photography. I feel Very few reviewers on YOUTUBE get this. I see reviews in the city like what the heck guys this camera is not the strongest in street photography probably the worse since it is So HUGE!!!\n\nOk, from the day the P1000 was announced I put up a post asking people if they were excited for the P1000. This post got crazy huge. Day 1 I pre-ordered it. After the upcoming weeks I started to question my order. There were some thing that scared me off from the camera. The questionable 3000mm zoom the lack of info on the new sensor and the list goes on an on.\n\nAnyway, I finally got the courage to drop 1K on it and it show up this morning. Part of me ordering the camera is there is so little info on how it works in wildlife for photography.\n\nNow I shoot wildlife with my P900 just as much as my DSLR gear. So after playing with the p1000 for the last few hours here are the quick improvements if you were concerned.\n\n1. Image quality ---> Yes better than the P900 , better noise, highlights, and sharpness is that or better. Under 2000mm for sure\n\n2. Ease of use, Yes more buttons manual focus rings, works faster, focus fast ect. Feels more like a DSLR.\n\n3. Buffer for shooting JPegs, Yes about 2 to 3 times more buffer it is what I needed the most for action photos. THIS was huge. I am not totally happy with the buffer but indeed much better.\n\n4. Build Quality--> Way better hands down,\n\nNIkon has made this for the WILDLIFE Hardcore... You can feel the direction they improved this camera. Higher build quality, more weight, It feels like if you really wanted to get serious about photography at almost a professional level this is what I would use. Durable like a DSLR NO but serious yes.\n\nLooking back at my photos 10 years ago with Nikon D80, d90 E500 and 500mm lenses, This beat them hands down in quality of photos and usability ect reach.\n\nIf you have any reservations on improvements do not be. They did a great job but my only concern is that it is made for TOO Serious of Photographers and will be considered a niche camera do to its size.\n\nIf you have any questions please ask. My self serving needs are I want more cameras like this versus my DSLR gear. I like DSLRS but they are super heavy in the 600 mm range.\n\nI might put out a Youtube Video comparing the differences??\n\nI think the reviews on this camera have been horrible so far. They concentrate on too many things as a general use camera. Again street photography is the last thing i would use it for.\n\nAny questions please ask\n\nThanks",
    "I was fortunate to buy a like-new camera from Japan. This is *not* an everyday camera, it's big and heavy, but I've taken photos of structures 11 miles away. And decent images of people a few miles away. There is even a built-in \"moon mode\" that set the focus at infinity and compensates for a bright moon filling most of the frame. For quality images from this tiny sensor, I stack several images to reduce noise and improve the detail. Distant shots are nearly always affected by thermal distortion, that's a given.\nFor walk-around shots, your current phone will probably give better results. For distant shots (and even photos of planets -- search the web) it's great. I'm certainly glad I purchased this niche camera, especially at a used price and with Amazon's return policy.\nI've attached a (stacked) photo of the Arizona Memorial from the U.S.S.Missouri",
    "Update October 12, 2018:\nI have completed four photo shoots since reporting the \u201chang up\u201d problem. The camera operated flawlessly as it should. No problems. Had to take videos and switch back to stills quickly, back and forth several times. Camera kept right up without missing a beat and stills look great. Usually there is a slight lull when going between videos and still shots. Pleased with operations. Another plus for the P1000. Changed rating back to 5 stars.\n\nUpdate October 1, 2018:\nMy P1000 \u201chung up\u201d on a photo field trip. After taking a photo, it would not focus or operate, or so it seemed. After a few seconds it took an exposure ( of the ground). This happened about four times after turning the power off then back on. Camera would operate fine then stop. I would wait to see if an exposure would happen. Sometimes it would. Checked self timer and it was off. Camera, zoom control, shutter would not work and only way to get camera operating again is to turn power off and on. I am thinking it is a software glitch. Anyone else have this problem? I have over 5K+ exposures on the camera and has worked flawlessly until now. Dropped my review from 5 to 4 stars.\nUpdate to original review:\nAdded couple shots of the moon. Saw the moon image done with ad for the P1000 and saw the full moon and wanted to see if the sample shown was for real. Well, here\u2019s my shots. What do you think? Larger shot done at max zoom (non digital to maintain sharpness).\n\nFirst off, I\u2019ve used my P900 for a couple of years on any photo shoot you can name. I\u2019ve used it as a video capture to single exposures. From sports to nature closeups. I am very happy with the P900.\nWhen I heard Nikon was coming out with the P1000, I put my order in for one. I got it in a week ago and read through the instructions, got it setup and ready. I have already taken it and my P900 to three photo shoots. Exposures were near perfect. I liked the extra bright viewfinder and how much clearer/sharper it is. The size is larger and heavier than the P900. At first it seemed a bit backwards in design but after using the camera I found the extra size and weight to help keep the camera steady on the longer zoom shots. The camera got alot of attention and most inquiries were positive once they got it in their hands. The menu is easy to use very similar to the P900. I especially appreciated the sunshade provided that was left off the P900 (Had to buy seperately). I also like the \u2018Hot Shoe\u2019 that was lacking on the P900. Another feature is ability to manual focus ring. Just flip a switch and focus away. If Nikon continues with cameras like the P1000, why would anyone need a DSLR? (Just kidding - I still love my DSLR Nikons!)\nFor an all around-do-it-all Point and Shoot shooting or videos, this one is a must. Recommeded.",
    "Let me state that I am an internationally published professional photographer and was an official contributor for Playboy SE Inc. for 5 years, so needless to say I have vast experience with cameras and DSLR\u2019s of all shapes and sizes. I shoot both Nikon and Canon and have all the standard array of top of the line lenses for both brands. While I always am using my full frame DSLR\u2019s over the past few years for lots of projects, one area of interest I\u2019ve been dedicating my spare time to is photographing the International Space Station as it flies overhead. It is visible to the naked eye at certain times just after dusk and just before dawn, and through apps you can predict when and where it will appear. It is by far the hardest thing I\u2019ve ever tried to shoot... a bright white object against a black background moving at 17,000 mph across the sky. I\u2019ve used all my zoom lenses, telescopes, more telescopes but it is hard to track with lots of extra gear and weight. I bought a P900 when it came out and was blown away at the capabilities of that camera. I never in a million years could imagine myself with a consumer brand Nikon CoolPix camera in my hand, but then again the CoolPix P900 did what none of my Professional Brand camera gear could do. However the main issue with the P900 and it\u2019s 2000mm equivalent zoom is that it only shot jpeg and someone like me shoots exclusively in raw NEF format. It\u2019s mandatory for me, but I settled in this case because its a trade off for the zoom factor.\nSo now the P1000 comes along, and not only have a much farther zoom capability, but this new model also shoots in Raw/NEF format! Oh the joy when I found out... and that\u2019s not all, this new P1000 has MANY upgrades from the P900. It is from the Coolpx family which Nikon markets to everyday consumers and such, but the P1000 is different. It\u2019s body is much bigger and sturdier than any other camera in the CoolPix line. It looks, feels, and pretty much performs like a higher end D750, except it has all the features that enable less experienced photographers take great pictures... things like scene mode, time lapses, it\u2019s so user friendly but has all the manual functions a camera snob like me demands. I\u2019ve seen guys in the field with those super high end telephoto lens that cost over 10k, but get you in close for wildlife and sports shooting... I always thought having gear like that\u2019d never be possible, but the new P1000 truly changes the playing field, for everyone. Kind of like the iPhone did for photography, this camera opens the impossible up for everyone now.",
    "I have been wanting a high zoom camera for a long time. I almost bought the P900, but didn\u2019t because of the lack of raw format. The P1000 looked to be just the thing. In the last couple months, I have been using it almost every day and it has renewed my interest in photography. While I love the zoom, there are a lot of things that took a back seat to achieving the incredible 125x zoom.\n\nPros\n\u2022 Crazy zoom. It can see things much farther than any lens I have owned. It has to be seen to appreciate how powerful the zoom is. You can\u2019t buy an equivalent lens for anything close to the price of this camera.\n\u2022 Vibration reduction is very good. You can take good shots at high zoom and not have camera shake.\n\u2022 Very good pictures with low zoom and good lighting\n\u2022 Can take good pictures at high zoom, but requires work.\n\u2022 Raw format\n\u2022 Saves carrying around a lot of equipment. Even though this camera is a little bulky, it is far lighter than my DSLR and the lenses.\nCons\n\u2022 Sensor is too small for this camera.\n\u2022 Max zoom requires target to be at least 21 feet away to focus on.\n\u2022 Lens does not gather enough light\n\u2022 AF does not work that great. I have many of slightly out of focus images.\no AF frequently fails at high zoom and/or low light\no AF can take almost a second to search\no Not suitable for fast moving subjects. AF is just too slow.\n\u2022 Not suited for low light setting. Forget high speed photography in anything but bright light.\n\u2022 Not suited to high shutter speed. Images often are too dark and cannot be fixed.\no Higher zoom limits shutter speed.\n\u2022 Image quality can be unacceptably poor at high zoom\n\u2022 Digital zoom has no value. Every picture I have taken with digital zoom, I have deleted because of image quality\n\u2022 Many throw away pictures. A lot of out of focus or too dark pictures that cannot be fixed in Photoshop.\n\u2022 Not all modes take RAW format image,\n\u2022 Snapbridge is very poor. They should farm it out to a 3rd party.\n\u2022 No protection from rain\n\u2022 You have to wait for the camera to zoom. You cannot zoom manually.\n\u2022 No outlet charger included.\n\u2022 Nikon batteries are expensive.\n\u2022 Distance to subject for reasonable quality, close up is at most 150 feet.\n\nGeneral observations/notes\n\u2022 Requires tripod for high zoom videos.\n\u2022 Needs a carrying case.\n\u2022 I recommend purchasing a second battery and an outlet charger.\n\u2022 I use a monopod and have not used the flash yet.\n\nI have attached a close up of a camel in bright light, using a tripod at full zoom. This is typical of this camera and demonstrates the poor quality typical of full zoom. I have included a shot farther away as a reference.\n\nEDIT: DO NOT USE A UV FILTER, IT REDUCES THE IMAGE QUALITY SIGNIFICANTLY.",
    "I bought this camera specifically for it's long focal-length video capability and had my doubts about what it could actually do. I've spent a LOT of money of long glass over the years and have a 1000mm AF f8 sitting in a closet to prove it. The claim that this thing was good to 3000mm optical was hard to believe but intriguing. After a month or so of use I have to say the camera is radically good. But the thing I like best about it might not be an advantage to you.\n\nAt extreme focal lengths the effect of the atmosphere becomes more and more apparent. Images shimmer and in some circumstances the air seems to boil. I love using this thing to shoot video of things moving -- cars, people walking, traffic -- and at 3000mm on a hot day, the effect is very beautiful but odd. I set this thing up on a tripod in the middle of the street and point it down the road to where a freeway on-ramp is a mile and a half away, and can clearly see vehicles make the turn at that distance. All the vehicles -- cars, trucks, bicycles, and pedestrians -- are stacked up between the camera and the on-ramp and they're all moving under their own power as well as in the boiling air. It is a very theatrical view.\n.\nThis is probably not what you're looking for if you're doing bird photography or all sorts of other work. If you're working closer, the shimmer would presumably be less.\n\nThe autofocus works reasonably well although it struggles to get a lock at long focal-lengths. It is somewhat slow but will follow-focus fairly well. Images are sharp and clean even at the end of the zoom.\n\nThe P1000 does not have many of the controls or features found on professional video cameras but it has enough that I'm happy with it. It works for me as an extreme-zoom lens with a fairly simple video camera attached.\n\nMy two cents, your mileage may vary, opinions void where prohibited.",
    "Pictures are from my P900, I expect better from my new P1000\nI see here and on You Tube reviewers talking about the weight of the camera, If you're use to a camera this big or heavy, you're going to keep on it a tripod.. that's a waste of a great camera. Two years ago before I got the P900 I used a Canon D5 Mark II and with a 24-105 len it was just as heavy and that was my go to camera for weddings, portraits, for my special events, sporting (NHRA drag racing) NFL football and college football games, nature, landscapes and many places where I can't use a tripod. So far I like this camera, I think many camera users are good photographers, they could be great if they stay away from the auto setting much as possible with a good camera as the P1000, I still plan to use my P900 when it calls for it, but I love shooting in raw which you can't with the P900. And a tip to getting great zoom photographs is using a tripod when shooting between 500 mm to 3000 mm, you're find that the photographs will be much clearer using the proper settings on the camera, yes the camera has a great stabilization to help . Tip for some photographers,,,, if you just take snap shots, drop down to a lower model....no need to spend $$$$$.00 When I do more testing I'll come back with P1000 photos and further reviews. So far in this early stage I'll give it 10 thumbs up",
    "I ran into a slight problem, in that I received a reconditioned product, albeit it was in totally 100% mint condition. If I had known the reconditioned model would be in such perfect condition, I probably would have opted for that cheaper option in the first place. Contacted seller, and he quickly offered a partial refund to cover the difference and promised to honor the full year standard warranty, if I should need that, (reconditioned models only offer 90 days). So I got a practically brand new product for cheaper, and coustomer service was quick, painless and a pleasure to deal with.\n\nThe camera itself is awesome. Zoom is unbelievable! While I'm not at all a professional it was pretty straightforward to use auto, point and shoot. Your going to want a tripod, but even without one, just resting my arms ok porch railing, I was able to get some stunning shots of the full moon last night. I'll get some shots of other things and upload a few of the best at some point. There are lots of features and I honestly haven't even really begun to figure them all out, but it's not too complicated, just gonna have to use auto whilst I overcome the learning curve. I don't expect to encounter many other observable curves on this p1000 journey, but I'm looking forward to collecting more and more first hand observations with this bad boy. Not supper cheap, but quality is too notch, you get what you pay for and in this case I'd say it's definitely worth it. Cheers",
    "I am so disappointed that I am writing this review. I just got home from the UPS store where I returned my new camera. Before ordering it I viewed all of the YouTube videos and the Nikon sales pitches. And I was sold. It looked amazing and I had been anticipating its arrival for weeks. Until I took it out of the box. It's huge. Not just big. HUGE. Cartoonish huge. It actually reminded me of my dad's camcorder from the 70s. The ones you had to mount on your shoulder. Obviously I was willing to accept it's size and wanted to see what it could do. I was smart enough to know that I would need a monopod to help me stabilize the camera, but it wasn't enough. I had a VERY difficult time keeping the lens on my subject when using the telephoto. I knew that it would be hard, but it was frustratingly difficult; to the point that I just started taking random pictures hoping that some of them would be framed properly.\n\nThe goods news: it does take great pictures. You just have to be able to deal with walking around with a five pound ham with a five foot stick poking out of it!\n\nJust fyi I have been taking photos for 59 years so I know a little about working a camera. I really am disappointed to be saying this. I wanted to give it five stars and rave about it, but I just can't. The other goods news is that I have another Nikon with three interchangeable lens so it looks like I'm sticking with that!\n\nAdditional comments after this was posted:\n\nWhy are so many of you taking this personally? I didn't attack your mother. I just was disappointed in this camera because it is too large and I don't want to always have to have a tripod or monopod when I'm shooting. It's not a bad camera. That's why I did not give it one or two stars. But I don't recommend it so I gave it 3 stars. It's my opinion. That's what these posts are supposed to be right?\n\nBy the way, that was really ignorant to say that I'm too old and weak for a big boy camera. I am not, but what if I was. What if I was 80 years old with multiple sclerosis? Would I be entitled to the opinion that this camera does not work for me? I get the feeling that you need to measure your manliness and virility by the size of your lens. Interesting.",
    "Camera seems fine, although has some lag with the auto focus. Initially thought I was sent an \"open box\" as the box was not sealed, manual was not in a plastic sleeve and the lens cap was also not wrapped. After reviewing some unboxing videos on YouTube, I think this is the way they're shipping them now. I would definitely prefer a seal on the box, better wrapping of the actual camera, and the manuals in plastic as they were rattling around in the box and made it sound like something was broken before I opened it. But it seems \"ok\" so far.",
    "I started taking pictures around 1970 with a Kodak Instamatic camera, which I still own, using 126 cartridge films. I now own several cameras; most of them are Nikon cameras. Every camera I own has its pros and cons. I'm not going to go through all the same things other reviewers have posted. I didn't buy this camera thinking I would get the same quality images I get from my Nikon D750 with a Nikon AF-S FX NIKKOR 24-70mm f/2.8E ED Vibration Reduction Zoom Lens. I bought the Nikon P1000 for all the pleasure I get from using it, sometimes I get good images and sometimes bad ones, as with most any camera. For the most part, the images I get from the P1000 have been very good. Yes, it\u2019s a big camera for a point and shoot but when its gets heavy I put it down. When it\u2019s raining I don\u2019t take it out to play, probably would be very careful at the beach because of sand and water. I saw a squirrel in the woods the other day and with the 3000mm zoom I got some great pictures without scaring him off. What I\u2019m trying to say is, if you can talk your wife into another new toy (a really great wife) and you have the money, buy the P1000, It\u2019s a lot of fun. I have owned the Nikon P1000 for\n\n1 Year; I'm still enjoying taking pictures with my Nikon P1000. I carry it in my car all the time, won't fit in my pocket like my cell phone. All the pictures shown are taken hand held including the moon. So far no problems with camera. Wife still questions the purchase :)) , but she likes all the pictures. This camera was purchased for fun and the interesting photos you can get with it.",
    "I have owned the Nikon P900 since they came out, and just received a P1000. I love the new P1000. Interestingly, I don't think the 3000mm maximum zoom is my favorite thing about it. Besides the obvious new features -- like the hot shoe, ability to use an external microphone, great eye-level viewfinder, and RAW shooting -- I really like the new control ring on the lens, which is very handy for manual focusing (especially with the new switch for quickly going from auto to manual focus). And the autofocus seems faster and more accurate. Lots of other new features that I am still learning to use but think will be very nice. And the menu system is much improved. This is a far better camera overall than the P900.\n\nYes, the P1000 is significantly larger and heavier than the P900, but it is not as heavy as a normal DSLR with a long lens. Neither the size or weight has been a problem. In fact, I think the extra weight (along with the larger grip) makes the P1000 more stable and less subject to shaking. All the attached photos were handheld and taken from a small boat. However, having said that, if they came out with a new version of the P900 that had the 2000mm lens but added all the other new features of the P1000, it would be a tough choice of which one I would prefer.\n\nBut in general, if you own a P900 and are wondering if you should upgrade to the P1000, I say go for it! So far, my only complaint is that the battery doesn't last as long as it does on the P900, so buy at least one extra battery.\n\nI'm looking into buying a dot sight to use for taking wildlife photos. Nikon is coming out with one, but my understanding is that any dot sight should work, and the Olympus EE-1 looks like a good choice. It has good reviews and is cheaper than the Nikon one (and is available now). Does anyone have experience with this?",
    "I am adding to my review below. It's been One year in a half of ownership. I see these guys all around with these slr's with HUGE lenses and a fraction of the power and feel sorry for them. I GET THE SHOT WAY OUT THERE! I'm still impressed with the quality. If Nikon produces one with 24mp I'll spend another grand and so forth on another one.\nBattery life nearly all day.\nSPEND MONEY on the fastest SD card possible. You won't regret it!\n(Original review)\nI am really impressed with\nthis camera. Not really heavy and not as bulky as I thought it would be. As a matter of fact I feel as though it is very lightweight. I'm finding I have better control with this size camera. The zoom lens is awesome! Does very well getting closer shots at reasonable distances. Very long distances does come with some atmospheric blur as expected. No fault to the camera however. One thing I am critical of is that I feel that Nikon could have upgraded the megapixel to make the image quality or comprable to a DSLR. That is why I only give image quality a 4-star rating. Many improvements. When I use the camera on burst mode all the images come out in perfect Focus. My p900 very much fell short this. I like the control ring around the lens of the camera. Can easily dial in features and especially exposure. I like that I can brighten and darken an image very quickly before taking the photo. I think this is a fantastic and easy camera for someone who is amateur like myself and works very well on automatic mode 4 nearly all scenes. I certainly recommend this camera to anyone who is looking for great quality and and extremely large Zoom range. I am extremely pleased with this purchase!",
    "I have own many Nikon cameras (and even some Canons) including the D850. I have purchased telephoto lenses for these DSLRs with disappointing results in their reach. This has been solved by a camera which costs a third of any of these telephoto lenses which would mount on my D850. The camera has the feel of DSLR but has a reach way beyond anything I have ever used before. The photographs are sharp and crop with clarity. I am amazed this camera is not priced more. Although this might not be the easiest camera to carry around during a trip, it will definitely be the best camera especially if you are planning on photographing landscape, animals or even sporting events, this is a camera which will give you some of the best photos you have ever taken. I would recommend using a tripod if you plan to use it at the extreme focal length. Otherwise this camera can easily be used handheld if you can handle the weight of a DSLR. I consider this camera one of the best buys I have made in photography over the last five years. Better than upgrading from the D750 to D850. This camera allows me to photograph objects I only dreamed about in the past since they were beyond the reach of my equipment/location. I now carry this camera with me instead of a $3000 to $4000 telephoto lens which does not even come close to this 3000 mm P1000 camera. If I need a zoom, I pull out this camera and keep a standard lens on my D850. The combination makes it possible for me to photograph anything, anywhere. Outstanding camera from its sensor to its lens. Can not imagine photography without this camera.",
    "I was so excited to read reviews on this camera and the zoom capabilities listed. Two months in and the wifi connection to SnapBridge doesn\u2019t function. I\u2019ve tried every troubleshoot technique available. I am returning and wish I had read reviews on Reddit before purchasing. It appears there are defects with Nikon software. I have a canon camera and have no issues with connecting to wifi to download pictures. One other issue is the camera gets out of focus too easily. I see other reviews mention this. What good is a zoom lens when you can\u2019t focus on your subject?",
    "So much time still learn on this camera but I love everything so far! Just make sure you get a American version and not a international version!!!!",
    "Since the user manual is printed in such small print I've got the pdf version up on my 55 inch monitor+ there are many getting started videos on the internet. Taking baby steps and will revisit this review in awhile after hands on use.",
    "I've had this camera for two days now and have worked thru the 175-page manual, so I feel safe posting some initial thoughts. First impression: it's BIG, and it's heavy. It is half-again bigger and heavier than my Lumix GH-4. But the Lumix doesn't have a 3000 mm zoom lens, which is the Nikon's raison d'\u00eatre. And that is where it shines. The sensor is small but the pictures are sharp. 24 mm wide angle and 3000 mm tele both produce nice images. (The photos I uploaded are 24, 1000, 2000, and 3000 mm, -- straight out of the camera. No tripod, just laying on a table, so the 3000 isn't as sharp as it could/should be.)\n\nThe menus are easy to navigate. The menu options are much more limited than the DH-4 but the most-used features are there, and are easily accessible. I would like to have more than one \"Fn\" button for quick access to menu items, but that's a small nit. There are physical buttons for exposure compensation, self-timer, focus mode, and flash settings. The manual focus works very easily, and I found it really handy at the long zoom settings. Overall, it's well done, with obvious design sacrifices for the enormous zoom range. I'm sure my DH-4 will remain my primary camera, but this Nikon seems to fulfill its main purpose quite well. (I took one star off for its bulk)\n\nSide note -- I'm not sure most people will find the extra zoom power over the earlier P900, which has a 2000mm tele, worth the several hundred dollar premium for the P1000. Plus the 1000 weighs an extra pound or so. What swung it for me was the RAW capabilities of the P1000, plus a few other features that a lot of folks might not care about.",
    "Many have complained of this camera being heavy. I've taken it out to take photos and enjoy the lesser weight than that of a 500mm or 600mm lens attached to a Nikon D7100 and get tremendously closer photos than I could with either of those lenses. The image stabilization works very well but you should still use a tripod when zooming out to the limits for a better shot. Remember to turn the stabilization off when using the tri-pod. The mechanics in the lens can cause motion even when on a tripod.\n\nI have taken amazing photos of the moon. At full extension of the lens, you only get a part of the moon. Craters are crisp and there are more visible than I had ever gotten with a large detachable lens. Have taken close shots and still be in focus that I never would have thought possible . It has a macro selection but it won't be needed unless you get extremely close.\n\nThere are some downers, i.e. total pixels and small sensor. Still, I have taken excellent photos with other cameras that were the same. The main reason I didn't like 16 mp was taking shots of birds that usually were a great distance away. This required cropping the photo which decreased the size of the file. With the P1000, no cropping is necessary for those distant birds. As a birder, this is a grand plus for me. Also, not having to change lenses for each situation is handy.\n\nEveryone has their own idea of what is ideal in a camera and lens. The P1000 fills a spot for the type of photography I do - wild birds and animals. One camera covers all situations.",
    "At long focal lengths, the P1000 gets much better pictures than a P900.\n\nUsing a tripod I got the attached sequence of a cormorant who was basking in the sun using purely optical telephoto at 24mm, 275mm, 3000mm. Using combined optical and electronic telephoto, on a tripod, I attach shots at 3600mm. 6000mm, 12000mm. I have a feeling that electronic telephoto is largely done by cropping.\n\nHand held, I attach pictures of a heron at 1000mm, a cormorant swimming at 2000mm and at 3000mm, all within about 200 feet of the camera. I also attach a hand held picture of a far subject, namely a cupola that was about 1500 feet away, at 3600mm. These were hand held without any support like a fence or a tree.\n\nSo this is a very capable camera, both on a tripod and, owing to good image stabilization, hand held.\n\nBattery life ? So far I have taken 291 pix, with lots of zooming in and out, and the battery indicator is now showing half depleted.\n\nAs backup for long trips I purchased two spare batteries and a charger for $18 (Powerextra 2x EN-EL20a Replacement Battery & Car Charger Compatible with Nikon Coolpix P1000). The charger also plugs into a 110 volt wall socket. Average review was 4.4 stars but several users warned that these batteries don't last all that long, so I'm going to keep the OEM battery in my camera bag and use the cheapies and see how well they do.\n\nTo protect the camera I bought an Amazon basics medium size camera bag which fits well, though would not protect against major impacts.",
    "These pictures are not the highest possible resolution for the camera. I'm having a great time with the P1000\nAlthough I had to have it replaced once, Amazon service was amazing. Shoot raw and the lower resolution of the camera is not an issue for me. JPEG only does not do what I enjoy, although is probably more than enough resolution for general posting sizes on line.\nThe \"digital extended zoom range is an issue and interesting. I have not used it enough to speak to it, beyond saying it may be an issue. My first one did not work at all. The replacement works fine, and works a little differently than the full manual suggests. First, be ready to select 'RESET TO FACTORY SETTINGS\" on the menu, even right out of the box, to get the digital zoom activated. Then, it is, as designed, available in some and not all shooting modes. The manual says \"not in bird mode,\" yet mine does have digital zoom access in Bird mode. I have not decided if it is a useful feature or not. The lens itself, autofocus, features of the camera, etc., are really fun! Buttons are well-located. I suggest ordering an extra battery or two, along with the quick charger. Otherwise, out of the box, charging is only while in the camera. The five stops of optical image stabilization is astounding. While a tripod with stabilizing off is nice, hand held and braced with stabilizing on is astonishing. I would buy this camera again.",
    "i got this camera last year, almost as an impulse buy - though i did watch a bunch of videos talking about it.\n\ni like this camera a lot, i have taken literally thousands of photos with it. its a professional looking point-n-shoot camera, and it does exactly what it says on the box. it takes really good photos most of the time, but i find that its hard to control everything.\n\nthe zoom is great. just not fun when on a tripod and you dont have a lens stabilizer. mine always falls down due to the weight. but when i get it work on the tripod, it takes amazing photos - especially of the moon. i also like to use it when taking photos of birds - i never have to get too close, but i find that the camera is a bit slow when the animals are moving.\n\ni like the camera, and ive had fun with it taking all my photos. ive recently upgraded though, and i feel like the p1000 will be on the shelf for a little bit before it sees use again.",
    "Clearly the zoom lens is the best feature. In my photo series you can see the ability to zoom in on a target well over 1KM away. You can almost read the brand of sunglasses the boat driver is wearing. The image stabilization is excellent. I also owned the Nikon P-900. Also an excellent camera. If Nikon releases an update to this camera I will buy it immediately.",
    "Unbelievable reach with its 3000mm lens. Persons will have trouble holding the camera still enough at max. range to avoid a slightly fuzzy picture in many cases. However it can be done with some practice. Great 4k vids and pics. Wonderful feel when holding this slightly larger than normal camera. Its heft and weight, however, I find are useful assets when filming at lens max. If you own a P900, you might want to think twice about this camera, however. The two are much too similar in most features. If you did not get a P900 because RAW was not included as part of that camera, this one has RAW for you purists. It is not an \"everything for every Photographer\" camera, but then, what camera is?? It is a Bridge Camera, not a DSLR, so you have both the advantages and disadvantages of that you get, and give up, in going to a Bridge Camera. If you do a lot of Nature Photography, or just live where you get to see a lot of nature out your back door, this is the camera for you!!! You will get those long distance pics of Mother Nature at her best that you would miss with another camera. And while I do wish it had better battery life, when out in Nature, that little failing is not enough to dampen my enthusiasm for this camera. I do recommend this camera.",
    "This is not a bad camera, true the pictures aren't all that good (16 MP ) at max zoom the photos are blurry. Now, the video capacity are outstanding. You won't find a better video camera unless you buy a professional camera ( thousands of $$$).\nI had to return mine because after I spent all that money, I got an open box ( or returned) camera.\nI'm kind of disappointed with Amazon.com because they are not doing a better job screening the sellers, this is becoming a \"flea market \". From now on ( after 20 years with Amazon) I will buy electronics from my local seller, that way I know I'm getting a brand new item.\nSo if you are looking for a good video camera, this is it ( pictures so, so )...\nThank you for reading.",
    "At first I was a little unsure about the weight and size, but after using it a few times and learning different settings, I love it. We mostly go birding and nature viewing. This is a great camera if you are taking a lot of variety of pictures from closeups of flowers, to family shots, to scenes and birds and animals and don\u2019t want to have to change lenses. It has a specific bird setting that works fantastic for capturing birds in trees and brush. The sports setting also worked great for capturing pictures of dolphins on a boat tour. The smart image setting is great as well. It isn\u2019t super fast at recovering for the next picture after taking a picture, especially after videos or multiple shots. I have a smaller camera I take places when I don\u2019t need the super zoom as it is quite bulky and heavy.",
    "This is my first take on the Nikon Coolpix P1000 camera. I'm an average novice photographer. Love taking pictures without putting a lot of major thought into it. The Nikon Coolpix P1000 has opened doors I never thought possible for me. As with all new cameras there's a learning curve but this one seems less than others.\n\nPROS: Love it! Despite being a bit on the heavy side( 3lbs 1.9 oz) the rewards are worth it. The pictures posted show why. Both were taken without a tripod. For the moonshot I used a porch post to lean against to stabilize the camera. Improvise..Adapt..Overcome!\n\nAs seen in the far right picture, the P1000 is about 1.5 times bigger than my D5500 with the Tamron 18mm-400mm lens.\n\nCONS: The camera didn't come with a separate wall charger. You have to charge the battery in the camera itself with the charger adapter kit provided. The battery is removeable and sits next to the SD card slot on the bottom of the camera.\n\nI am not sure someone with small hands could hold this camera comfortably and would most likely have to resort to a monopod or tripod. I'm a big guy so it was no problem.\nIt is a bit on the pricey side but aren't you worth it? You can't put a price on the fun you'll have with this great camera..",
    "If you are looking into this camera it is probably because of the 3000mm built-in zoom and macro capability. I have used this camera on a couple of big trips-The Galapagos Islands and a recent Italy & Greece trip. I have found that it really beats carrying around extra lenses. The weight is tolerable for a full day of shooting if you support the camera when not taking pictures or else use a shoulder strap.\nWhile the extreme zoom can present problems with noise and framing issues under cloudy or gray conditions, it is excellent on a clear, sunny day. The photos I have attached of the Hephaestus Temple that I shot while descending the Acropolis, are a good example of this cameras zoom capability on a perfect day. Except for reducing these images in size for the web, they are not re-touched. I find this camera to be a great solution for people who don't want to carry a lot of equipment.",
    "I bought my last Nikon camera ten years ago. I bought this one because I wanted better moon photos. When I extended the lens all the way to 3000 mm on my original tripod, the tripod almost went face-first onto the ground. (I fixed that problem with a new Manfrotto tripod.) Once you have the right tripod, or you have extremely steady hands, your moon pictures will be the envy of the neighborhood. The next morning, if you want to take close-up pictures of your flowers, you'll use this camera in macro mode at 1 centimeter away. Purists will say you could have a bigger sensor, or more lens control, or blah blah blah. Tell them to go jump in the lake. This is a \"bridge camera;\" it delivers superb pictures up close, and it delivers mind-blowing moon pictures, all without stopping to change lenses. P.S.: I also record training videos with this camera. Did I mention it can record at 1080 resolution and in 4K resolution? I added an Audio-Technica shotgun mike to the accessory shoe on the camera, to record my granddaughter's orchestra concerts. Stop reading, and buy your own Nikon Coolpix P1000. You'll be glad you did.",
    "no review",
    "Wow, the zoom; it is incredible. And the EVF is really nice, bright and sharp. I think that about sums up the pros. I have to say that I really wanted to like this camera, I really did. I knew that it was large and heavy, and it is, but I knew that going into it. The two deal killers for me, and the reasons I ended up sending it back, is because the AF is super inconsistent beyond 1500 mm and the image quality beyond 1500 mm degraded quite a bit. The images at 3000 mm is more useful for snap shots or memory sakes but they wasn't anything I'd consider printing or hanging up.\n\nWhenever I started zooming out, the AF failed quite a bit and I had to switch to manual focusing, and this isn't very helpful if the target moves much. To make matters worse the manual focusing ring was slow to adjust the focus. I could almost live with this if the image quality was good that far out, but it wasn't and that was the biggest disappointment. All that zoom is wasted if the images aren't satisfactory.\n\nI had the P900 before and I think it was a better camera. The main reasons I wanted this camera was the extra 1000 mm of zoom and the hot shoe, but the P900 just took better pictures.",
    "just love the zoom",
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

def find_original_word(feature_name, review):
    if feature_name == 'easi':
        feature_name = 'easy'
    if feature_name == 'mani':
        feature_name = 'many'
    pattern = r'\b\w*' + re.escape(feature_name) + r'\w*\b'
    match = re.search(pattern, review, re.IGNORECASE)
    if match:
        return match.group()
    else:
        return feature_name

def ExtractAspects(reviews):
    new_reviews = Preprocessing(reviews)
    doc = nlp(' '.join(new_reviews)) # convert list of reviews to string (nlp only accepts string)
    aspects = set() # set() to remove duplicates
    lemmatizer = WordNetLemmatizer()
    for token in doc:
        if token.pos_ == 'NOUN' and token.dep_ == 'nsubj':
            if len(token.text) > 3 and token.text not in stops: # exclude single-character words and stopwords
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
                count += len(re.findall(r'\b{}\b'.format(aspect), review, re.IGNORECASE))
        aspect_counts[aspect] = count
    # most_mentioned = aspect_counts.most_common(6) # get the 6 most mentioned aspects with their counts (uncomment to check for aspect frequency)
    # print(most_mentioned) # print the 6 most mentioned aspects with their counts
    top_aspects = [aspect for aspect, count in aspect_counts.most_common(6)]
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
    # print(json.dumps(aspect_sents, indent=2))
    return aspect_sents

# def split_text_sliding_window(text, max_length=1024, step_size=512):
#     text_chunks = []
#     start = 0
#     end = max_length
#     while start < len(text):
#         text_chunks.append(text[start:end])
#         start += step_size
#         end += step_size
#     return text_chunks


# def summarize_chunk(chunk):
#     return summarizer(chunk, min_length=10, max_length=50)[0]['summary_text']

def summarize_text(aspect_sents, summarizer_model, tokenizer):
    summaries = {}
    for aspects, sents in aspect_sents.items():
        random.shuffle(sents)
        tokens = tokenizer.encode(' '.join(sents), return_tensors='pt') # max_length=1024, truncation=True
        summary_ids = summarizer_model.generate(tokens, max_length=80, num_beams=2, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        if not summary.endswith('.'):
            # Remove the last sentence
            summary = '. '.join(summary.split('. ')[:-1]) + '.'
            summaries[aspects] = summary
            # print(f"Aspect: {aspects}")
            # print(f"\t\t{summary}\n")
    print(json.dumps(summaries, indent=2))
    return summaries

# def generateDescription(product_name, generator, top_aspects):
#     for aspect in top_aspects:
#         prompt = f'Generate a description for {aspect} in relation to {product_name}.'
#         description = generator(prompt, max_length=100)[0]['generated_text']
#         print(description)

def getSentiment(reviews):
    # vectorizer = TfidfVectorizer()
    model = pickle.load(open("SentimentModel/modelCraig.pkl", 'rb'))
    # print(model)
    aspect_sentiments = []
    for review in reviews:
        lemma = WordNetLemmatizer() # Instantiate PorterStemmer
        letters_only = re.sub("[^a-zA-Z]", " ", review) # Remove non-letters
        words = letters_only.lower().split() # Convert words to lower case and split each word up
        global stops # Remove stopwords
        meaningful_words = [w for w in words if w not in stops]
        meaningful_words = [lemma.lemmatize(w) for w in meaningful_words] # Stem words
        # Join words back into one string, with a space in between each word
        final_text = pd.Series(" ".join(meaningful_words))
        # X_vec = vectorizer.fit_transform(final_text)
        # Generate predictions
        pred = model.predict(final_text)[0]
        proba = model.predict_proba([pd.Series.to_string(final_text)])[0]
        # pred = model.predict(X_vec)[0]
        # proba = model.predict_proba(X_vec)[0]

        positive_prob = proba[0]
        negative_prob = proba[1]
        overall_prob = 0
        if pred == 1:
            output = "Negative"
            overall_prob = negative_prob
        else:
            output = "Positive"
            overall_prob = positive_prob

        for clf in model.estimators_:
            if isinstance(clf, GridSearchCV):
                best_estimator = clf.best_estimator_
                if isinstance(best_estimator, Pipeline):
                    nb_clf = best_estimator.named_steps['nb']
                    vectorizer = best_estimator.named_steps['tvec']
                    # get log probabilities of features given each class
                    log_prob = nb_clf.feature_log_prob_
                    # convert log probabilities to regular probabilities
                    prob = np.exp(log_prob)
                    # get feature names
                    feature_names = vectorizer.get_feature_names_out()
                    # get probabilities for each word in review
                    pos_word_probs = {}
                    neg_word_probs = {}
                    for feature_name, pos_feature_prob, neg_feature_prob in zip(feature_names, prob[0], prob[1]):
                        if feature_name in meaningful_words:
                            pos_word_probs[feature_name] = pos_feature_prob
                            neg_word_probs[feature_name] = neg_feature_prob
                            if output == 'Positive':
                                most_positive_word = max(pos_word_probs, key=pos_word_probs.get)
                                new_entry = {  # Declare and initialize new_entry
                                    'sentence': review,
                                    'label': output,
                                    'probability': overall_prob,
                                    'word_proba': find_original_word(most_positive_word, review),
                                }
                                if not any(e['sentence'] == review for e in aspect_sentiments):
                                    aspect_sentiments.append(new_entry)
                            elif output == 'Negative':
                                most_negative_word = max(neg_word_probs, key=neg_word_probs.get)
                                new_entry = {  # Declare and initialize new_entry
                                    'sentence': review,
                                    'label': output,
                                    'probability': overall_prob,
                                    'word_proba': find_original_word(most_negative_word, review)
                                }
                                if not any(e['sentence'] == review for e in aspect_sentiments):
                                    aspect_sentiments.append(new_entry)
    # print(aspect_sentiments)
    return aspect_sentiments


def analyzeAspectPhrases(aspect_sents):
    aspect_sentiments = {}
    for aspect, sentences in aspect_sents.items():
        positive_reviews = []
        negative_reviews = []
        aspect_sentiments[aspect] = {}
        sentiments = getSentiment(set(sentences))
        for sentiment in sentiments:
            if sentiment['label'] == 'Positive':
                positive_reviews.append(sentiment)
            else:
                negative_reviews.append(sentiment)
        aspect_sentiments[aspect]['Positive'] = positive_reviews
        aspect_sentiments[aspect]['Negative'] = negative_reviews
        # print(f"Aspect: {aspect} \n")
        # print(f"Positive Reviews: {len(positive_reviews)}, ")
        # print(f"Negative Reviews: {len(negative_reviews)}")
    # print(json.dumps(aspect_sentiments, indent=2))
    return aspect_sentiments

from nltk.tokenize import sent_tokenize

def analyzeAllReviews(reviews):
    analyzed_reviews = {}
    analyzed_reviews['Positive'] = []
    analyzed_reviews['Negative'] = []
    tokenized_reviews = [sent for review in reviews for sent in sent_tokenize(review)]
    analysis = getSentiment(tokenized_reviews)
    for a in analysis:
        if a['label'] == 'Positive':
            analyzed_reviews['Positive'].append(a)
        else:
            analyzed_reviews['Negative'].append(a)
    # print(f"Positive reviews: {len(analyzed_reviews['Positive'])}")
    # print(f"Negative reviews: {len(analyzed_reviews['Negative'])}")
    
    # print(json.dumps(analysis, indent=2))
    return analysis



def getRawSentimentScore(phrases):
    sentiment_counts = {}
    for aspect, sentences in phrases.items():
        sentiment_counts[aspect] = {'Positive': 0, 'Negative': 0}
        sentiment = getSentiment(sentences)
        # print(json.dumps(sentiment, indent=2))
        # Count the number of positive and negative reviews
        counts = Counter([s['label'] for s in sentiment]) # 'label' is Positive or Negative
        # Update the sentiment_counts dictionary
        sentiment_counts[aspect]['Positive'] = counts['Positive']
        sentiment_counts[aspect]['Negative'] = counts['Negative']
    # print(json.dumps(sentiment_counts, indent=2))    
    return sentiment_counts

def getTotalSentimentCounts(raw_sentiment_score):
    total_positive = sum([counts['Positive'] for counts in raw_sentiment_score.values()])
    total_negative = sum([counts['Negative'] for counts in raw_sentiment_score.values()])
    # print(f"Total Positive: {total_positive}")
    # print(f"Total Negative: {total_negative}")
    return {'Positive': total_positive, 'Negative': total_negative}

        
def getNormalizedSentimentScore(phrases):
    normalized_counts = {}
    for aspect, sentences in phrases.items():
        normalized_counts[aspect] = {}
        sentiment = getSentiment(sentences)
        positive_proba = (s['probability'] for s in sentiment if s['label'] == 'Positive')
        negative_proba = (s['probability'] for s in sentiment if s['label'] == 'Negative')
        # print(len(list(positive_proba))) # uncomment to check if positive_proba is equal to Positive counts in getRawSentimentScore
        # print(len(list(negative_proba))) # uncomment to check if negative_proba is equal to Negative counts in getRawSentimentScore
        positive_list = list(positive_proba) # convert to list to avoid 'generator object' error
        negative_list = list(negative_proba) # convert to list to avoid 'generator object' error
        mean_positive = sum(positive_list) / len(positive_list) if positive_list else 0
        # print(mean_positive)
        # if else statement above is to avoid ZeroDivisionError
        mean_negative = sum(negative_list) / len(negative_list) if negative_list else 0
        # print(mean_negative)
        most_positive = next(s for s in sentiment if s['probability'] == max(positive_list))
        most_negative = next(s for s in sentiment if s['probability'] == max(negative_list)) if negative_list else 'None'
        most_positive_word = find_original_word(most_positive['word_proba'], most_positive['sentence'])
        most_negative_word = find_original_word(most_negative['word_proba'], most_negative['sentence']) if most_negative != 'None' else 'None'
        normalized_counts[aspect]['Most_Positive_Sentence'] = most_positive['sentence']
        normalized_counts[aspect]['Positive_Probability'] = (most_positive['probability'] * 100)
        normalized_counts[aspect]['Positive_Word'] = most_positive_word
        normalized_counts[aspect]['Most_Negative_Sentence'] = most_negative['sentence'] if most_negative != 'None' else 'None'
        normalized_counts[aspect]['Negative_Probability'] = (most_negative['probability'] * 100) if most_negative != 'None' else 'None'
        normalized_counts[aspect]['Negative_Word'] = most_negative_word
        
        # if mean_positive > mean_negative: 
        #     overall_sentiment = 'Positive'
        #     mean_proba = round(mean_positive * 100, 2)
        #     normalized_counts[aspect]['Normalized_Sentiment'] = overall_sentiment
        #     normalized_counts[aspect]['Normalized_Proba'] = str(mean_proba) + '%'
        #     # most_positive = next(s['review'] for s in sentiment if s['probability'] == max(positive_list))
        #     # print(most_positive.encode('utf-8', errors='replace'))
        # elif mean_negative > mean_positive:
        #     overall_sentiment = 'Negative'
        #     round(mean_negative * -100, 2)
        #     normalized_counts[aspect]['Normalized_Sentiment'] = overall_sentiment
        #     normalized_counts[aspect]['Normalized_Proba'] = str(mean_proba) + '%'
        # else:
        #     overall_sentiment = 'Neutral'
        #     mean_proba = 0
        #     normalized_counts[aspect]['Normalized_Sentiment'] = overall_sentiment
        #     normalized_counts[aspect]['Normalized_Proba'] = str(mean_proba) + '%'

    # print(json.dumps(normalized_counts, indent=2))
    return normalized_counts

# def summarize_text(aspect_phrases):
#     summary = {}
#     for aspect, sentences in aspect_phrases.items():
#         text = ' '.join(sentences)
#         prompt = f"Write a summary of the following {aspect} reviews: {text}"
#         input_ids = tokenizer.encode(prompt, return_tensors='pt')
#         output = model.generate(input_ids, max_length=50)
#         summary[aspect] = tokenizer.decode(output[0], skip_special_tokens=True)
#     return summary

if __name__ == '__main__':
    aspects = ExtractAspects(reviews)
    top_aspects = ExtractTopAspects(reviews, aspects)
    sentiments = getSentiment(reviews)
    aspect_phrases = ExtractAspectPhrases(reviews, top_aspects)
    # print(json.dumps(aspect_phrases, indent=2))
    # product_name = 'Nikon Coolpix P1000'
    # generateDescription(product_name, generator, top_aspects)
    summarize = summarize_text(aspect_phrases, summarizer_model, tokenizer)
    # print(json.dumps(summarize, indent=2)) 
    # print(json.dumps(aspect_phrases, indent=2))
    # aspect_sentiments = analyzeAspectPhrases(aspect_phrases)
    # raw_score = getRawSentimentScore(aspect_phrases)
    # normalized_score = getNormalizedSentimentScore(aspect_phrases)
    # review_analysis = analyzeAllReviews(reviews)
    # total_score = getTotalSentimentCounts(raw_score)
    # summarize_phrases = summarize_text(aspect_phrases)
    # print(json.dumps(summarize_phrases, indent=3))
    
# print(json.dumps(normalized_score, indent=2))
# Call the functions and store the results in a dictionary
# results = {}
# results['aspects'] = ExtractAspects(reviews)
# results['top_aspects'] = ExtractTopAspects(reviews, results['aspects'])
# results['aspect_phrases'] = ExtractAspectPhrases(reviews, results['top_aspects'])
# results['raw_score'] = getRawSentimentScore(results['aspect_phrases'])
# results['normalized_score'] = getNormalizedSentimentScore(results['aspect_phrases'])

# Write the results to a JSON file
# with open('output.json', 'w') as f:
#     json.dump(results, f, indent=2)