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


from nltk.classify.naivebayes import NaiveBayesClassifier
from xml.etree import ElementTree

import pickle
import csv
import json
import sys
import csv
import pandas as pd
import numpy
from sklearn.svm import SVR


tknzr = TweetTokenizer()


reload(sys)
sys.setdefaultencoding('ISO-8859-1')


def extract_features(document, word_features):

    document_words = set(document)

    features = {}

    for word in word_features:

        features['contains(%s)' % word] = (word in document_words)

    return features

def get_word_features(wordlist):

    wordlist = FreqDist(wordlist)

    word_features = wordlist.keys()

    return word_features


def get_tweetsword(tweets):

    all_words = []

    for (words, sentiment) in tweets:

      all_words.extend(words)

    return all_words

def predict_and_write(Genderclassifier, Ageclassifier, test_data_dir, output_dir, Age_word_features, Gender_word_features):
    if not os.path.exists(output_dir):
        #Making output directory if doenot exist
        os.makedirs(output_dir)
        #Loading the Personlaity Open Pickle file
    file1 = open("/home/itadmin/Personality/P_ope.pickle", 'rb')
    Opeclassifier = pickle.load(file1)
    file1.close()
    #Loading the Personlaity Agreeable Pickle file
    file2 = open("/home/itadmin/Personality/P_agr.pickle", 'rb')
    Agrclassifier = pickle.load(file2)
    file2.close()
    #Loading the Personlaity Conscientious Pickle file
    file3 = open("/home/itadmin/Personality/P_con.pickle", 'rb')
    Conclassifier = pickle.load(file3)
    file3.close()
    #Loading the Personlaity Extrovert Pickle file
    file4 = open("/home/itadmin/Personality/P_ext.pickle", 'rb')
    Extclassifier = pickle.load(file4)
    file4.close()
    #Loading the Personlaity Neurotic Pickle file
    file5 = open("/home/itadmin/Personality/P_neu.pickle", 'rb')
    Neuclassifier = pickle.load(file5)
    file5.close()
    #loading the good.text file which contain positive
    pos_sent = open("/home/itadmin/good.txt").read()
    positive_words = pos_sent.split('\n')

    #loading the bad.text file which contain negative
    neg_sent = open("/home/itadmin/bad.txt").read()
    negative_words = neg_sent.split('\n')

    profile_file_path = os.path.join(test_data_dir, "profile/profile.csv")
    print profile_file_path
    with open(profile_file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            uuid = row['userid']
            #Act_gender=row['gender']
            #Act_age=row['age']
            positive =    0
            negative = 0
            neutral = 0
            df = pd.DataFrame()

            df['positive'] = 0
            df['negative'] = 0
            df['neutral'] = 0
            test_file = os.path.join(test_data_dir, "text", uuid+".txt")
            with codecs.open(test_file, "r", "ISO-8859-1") as fo:
                try: 
                    temp = []
                    for line2 in fo:
                        stopword = stopwords.words('english')
                        for word in tknzr.tokenize(line2):
                    	    temp2 = word.lower()
                       	    
                            if temp2 not in stopword:
                                
                                temp.append(temp2)
                            	if temp2 in positive_words:
			            positive = positive + 1
                                    
                                elif temp2 in negative_words:
                                    negative = negative + 1
                                else :
                                    neutral = neutral + 1
                    df.set_value(0, 'positive', positive)
                    df.set_value(0, 'negative', negative)
                    df.set_value(0, 'neutral', neutral)

                    #Predicting the Personality Traits Values
                    neu = str(int(round(Neuclassifier.predict(df)[0], 0)))
                    ope = str(int(round(Opeclassifier.predict(df)[0], 0)))
                    con = str(int(round(Conclassifier.predict(df)[0], 0)))
                    ext = str(int(round(Extclassifier.predict(df)[0], 0)))
                    agr = str(int(round(Agrclassifier.predict(df)[0], 0)))
                   
                    gender = Genderclassifier.classify(extract_features(temp, Gender_word_features))
                    age = Ageclassifier.classify(extract_features(temp, Age_word_features))
                    print("Gender Prediction ", gender)
                    # Confusion matrix for Gender Prediction
                    #confusion_matrix = ConfusionMatrix(Act_gender, gender)
                # print("Confusion matrix:\n%s" % confusion_matrix)
		    print("Age Prediction ", age)
                #  Confusion matrix for Age
                 #confusion_matrix = ConfusionMatrix(, attrs[gender])
                # print("Confusion matrix:\n%s" % confusion_matrix)
                except UnicodeDecodeError: 
		    gender = "0"
            #confusion_matrix = ConfusionMatrix(, attrs[gender])
                # print("Confusion matrix:\n%s" % confusion_matrix)
                    #Saving the Predictions in the output
                output_file = os.path.join(output_dir, uuid+".xml")
                with open(output_file, "w") as out_f:
                    attrs = {"\n userId": uuid,
                             "\n gender" : "male" if gender == "0" else "female",
                             "\n age_group" : age,
                             "\n extrovert" : ext,
                             "\n neurotic" : neu,
                             "\n agreeable" : agr,
                             "\n conscientious" : con,
                             "\n open" : ope
                             }

                    tree = ElementTree.Element("user", attrs)
                    out_f.write(ElementTree.tostring(tree))
                  

def run(test_dir, output_dir):
    #Loading the Gender Classification Pickle file
    Pick1 = open("/home/itadmin/Gender_C.pickle", 'rb')
    Genderclassifier = pickle.load(Pick1)
    Pick1.close()
    #Loading the Age Classifier Pickle File
    Pick2 = open("/home/itadmin/Age_C.pickle", 'rb')
    Ageclassifier = pickle.load(Pick22)
    Pick2.close()
    print ("works")
    with open("/home/itadmin/Age_T.pickle", 'rb') as Ageresults:
    	Agetweets = pickle.load(Ageresults)
    Ageresults.close()
    with open("/home/itadmin/Gender_T.pickle", 'rb') as Genderresults:
    	Gendertweets = pickle.load(Genderresults)
    Genderresults.close()
    Age_word_features = get_word_features(get_tweetsword(Agetweets))
    Gender_word_features = get_word_features(get_tweetsword(Gendertweets))

    #Callingthe Predict_and _write Function
    predict_and_write(Genderclassifier, Ageclassifier, test_dir, output_dir, Age_word_features, Gender_word_features)


def main(args):
   run(args.test_dir, args.output_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="""Script takes full input path to
                         test directory, output directory and training directory""")

# For Getting the Input and Output Directory
    parser.add_argument('-i',
                        "--test_dir",
                        type=str,
                        required=True,
                        help='Full path to input test directory containing profile and text dir')

    parser.add_argument('-o', "--output_dir",
                        type=str,
                        required=True,
                        help='The path to output directory')


    args = parser.parse_args()
    return args

#Main
if __name__ == "__main__":
    args = parse_args()
    main(args)



