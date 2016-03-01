# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 07:40:16 2016

@author: abcd
"""
#!/usr/bin/python
import argparse
import io

import os
import re

import codecs
import traceback
import logging
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

#from textblob import formats
from nltk.classify.naivebayes import NaiveBayesClassifier
from xml.etree import ElementTree

import pickle
import csv
import json
import sys
import csv

tknzr = TweetTokenizer()

reload(sys)
sys.setdefaultencoding('ISO-8859-1')

def get_wtweets(tweets):

    all_words = []

    for (words, sentiment) in tweets:

      all_words.extend(words)

    return all_words


#
def extract_features(document, word_features):

    document_words = set(document)

    features = {}

    for word in word_features:

        features['contains(%s)' % word] = (word in document_words)

    return features

def get_wfeatures(wordlist):

    wordlist = FreqDist(wordlist)

    word_features = wordlist.keys()

    return word_features

def predict_and_write(Genderclassifier, Ageclassifier, test_data_dir, output_dir, word_featuresAge, word_featuresGender):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    profile_file_path = os.path.join(test_data_dir, "profile/profile.csv")
    print profile_file_path
    with open(profile_file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            uuid = row['userid']
            test_file = os.path.join(test_data_dir, "text", uuid+".txt")
            with open(test_file, "rb") as fo:
                try: 
                    temp = []
                    for line2 in fo:
                        linetemp = line2.encode('utf-8').strip()
                        for word in tknzr.tokenize(line2):
                    	    temp2 = word.encode('utf-8')
                       	    temp.append(temp2)
                    gender = Genderclassifier.classify(extract_features(temp, word_featuresGender))
                    age = Ageclassifier.classify(extract_features(temp, word_featuresAge))
                    print("Gender Prediction ", gender)
		    print("Age Prediction ", age)
                except UnicodeDecodeError: 
		    gender = "0"
                output_file = os.path.join(output_dir, uuid+".xml")
                with open(output_file, "w") as out_f:
                    attrs = {'userId': uuid,
                             'gender' : "male" if gender == "0" else "female",
                             'age_group' : age,
                             'extrovert' : "1",
                             "neurotic" : "1",
                             "agreeable" : "1",
                             "conscientious" : "1",
                             "open" : "1"
                             }
                    tree = ElementTree.Element('', attrs)
                    out_f.write(ElementTree.tostring(tree))


def run(test_dir, output_dir):
    f1 = open("/home/itadmin/Gender_classifier2.pickle", 'rb')
    Genderclassifier = pickle.load(f1)
    f1.close()
    f2 = open("/home/itadmin/Age_classifier2.pickle", 'rb')
    Ageclassifier = pickle.load(f2)
    f2.close()
    print ("works")
    with open("/home/itadmin/AgeTweets.pickle", 'rb') as Ageresults:
    	Agetweets = pickle.load(Ageresults)
    Ageresults.close()
    with open("/home/itadmin/GenderTweets.pickle", 'rb') as Genderresults:
    	Gendertweets = pickle.load(Genderresults)
    Genderresults.close()
    word_featuresAge = get_wfeatures(get_wtweets(Agetweets))
    word_featuresGender = get_wfeatures(get_wtweets(Gendertweets))
    
    predict_and_write(Genderclassifier, Ageclassifier, test_dir, output_dir, word_featuresAge, word_featuresGender)




def main(args):
   run(args.test_dir, args.output_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="""Script takes full input path to
                         test directory, output directory and training directory""")


    parser.add_argument('-i', "--test_dir",
                        type=str,
                        required=True,
                        help='Full path to input test directory containing profile and text dir')

    parser.add_argument('-o', "--output_dir",
                        type=str,
                        required=True,
                        help='The path to output directory')


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)



