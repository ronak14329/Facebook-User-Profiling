# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 07:08:26 2016

@author: ronak
"""

import os
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import pickle
from nltk import FreqDist
import nltk
import json

tknzr = TweetTokenizer()
#  
tweets= []
with open("C:/Users/abcd/Documents/Train/Profile/Profile.csv", "r") as profiles:
    next(profiles)
    for line in profiles:
        main = re.split(',', line)
        userID = main[1]
        age = float(main[2])
        agegroup = " "
        if age<=24:
            agegroup = "xx-24"
        elif age>24 and age<=34:
            agegroup = "25-34"
        elif age>34 and age>=49:
            agegroup = "35-49"
        else:
            agegroup = "50-xx"
            
        list_of_words = []
        with open("C:/Users/abcd/Documents/Train/Text/"+userID+".txt", "r") as status:
            for line2 in status:
                stop_words  = stopwords.words('english')
                for i in tknzr.tokenize(line2):
                    temp = i.encode(encoding='utf-8', errors='ignore').lower()
                    if temp not in stop_words:
                        if temp not in list_of_words:
                            list_of_words.append(temp)
        tweets.append((list_of_words,agegroup))
#         
with open("C:/Users/abcd/Documents/Train/AgeTweets.pickle", 'wb') as resultsFile:
    pickle.dump(tweets, resultsFile, protocol = 2)
 
print (len(tweets))

with open("C:/Users/abcd/Documents/Train/AgeTweets.pickle", 'rb') as results:
    tweets2 = pickle.load(results)

def get_words_in_tweets(tweets):
   
    all_words = []
   
    for (words, sentiment) in tweets:
   
      all_words.extend(words)
   
    return all_words
   
def get_word_features(wordlist):
   
    wordlist = FreqDist(wordlist)
   
    word_features = wordlist.keys()
   
    return word_features
 
word_features = get_word_features(get_words_in_tweets(tweets2))

# 
# 
def extract_features(document):
  
    document_words = set(document)
  
    features = {}
  
    for word in word_features:
  
        features['contains(%s)' % word] = (word in document_words)
  
    return features
 
training_set = nltk.classify.apply_features(extract_features, tweets2)     
print (len(training_set))
 
Classifier = nltk.NaiveBayesClassifier.train(training_set)
 
f = open("C:/Users/abcd/Documents/Train/Age_classifier.pickle", 'wb')
pickle.dump(Classifier, f, protocol = 2)
f.close()
print ("works")

 f = open("C:/Users/abcd/Documents/Train/Age_classifier.pickle", 'rb')
 classifier = pickle.load(f)
 f.close()
 c = classifier.classify(extract_features("This is a fantastic api!".split()))
 print ("Classify: {}".format(c))

# count = 0
# with open("C:/Users/abcd/Documents/Train/testprofiles.csv", "r") as profiles:
#     next(profiles)
#     for line in profiles:
#         main = re.split(',', line)
#         userID = main[1]
#         age = int(main[2])
#         agegroup = ""
#         if age<=24:
#             agegroup = "xx-24"
#         elif age>24 and age<=34:
#             agegroup = "25-34"
#         elif age>34 and age>=49:
#             agegroup = "35-49"
#         else:
#             agegroup = "50-xx"

#         list_of_words = []
#         with open("C:/Users/abcd/Documents/Train/text/"+userID+".txt", "r") as status:

#             temp = []
#             for line2 in status:
#                 for word in tknzr.tokenize(line2):
#                     temp2 = word.encode(encoding='utf_8', errors='strict')
#                     temp.append(temp2)
#             c = classifier.classify(extract_features(temp))


#             if c == agegroup:
#                 count = count + 1


 #print (count)


 # count = 0
 # with open("C:/Users/abcd/Documents/Train/text/404ad905d2feec0f6b0984df9233e321.txt", "r") as status:
 #    temp = []
 #    for line2 in status:
 #        for word in tknzr.tokenize(line2):
 #            temp2 = word.encode(encoding='utf_8', errors='strict')
 #            temp.append(temp2)
 #    c = classifier.prob_classify(extract_features(temp)).max()
 #    print ("Classify: {}".format(c))
 #    print (c)
    ###