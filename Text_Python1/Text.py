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
    #Reading the good.text file which contain positive
    pos_sent = open("/home/itadmin/good.txt").read()
    positive_words = pos_sent.split('\n')

    #Reading the bad.text file which contain negative
    neg_sent = open("/home/itadmin/bad.txt").read()
    negative_words = neg_sent.split('\n')

#///////////READING THE TEST DATA///////////////////

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
            
#//////////////INITIALIZING THE VALUE TO THE DATAFRAME///////////

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
                  
# /////////THIS FUNCTION IS UPDATED TO USE NRC LEXICON EMOTIONS FOR PERSONALITY TRAITS/////////////

#def predict_and_write(Genderclassifier, Ageclassifier, test_data_dir, output_dir, word_featuresAge, word_featuresGender):
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
# ///////////////////LOADING ALL THE REQUIRED PICKLE FILE IN THE PROGRAM////////////
#    f1 = open("/home/itadmin/NRC_ope.pickle", 'rb')
#    Opeclassifier = pickle.load(f1)
#    f1.close()
#    f2 = open("/home/itadmin/NRC_agr.pickle", 'rb')
#    Agrclassifier = pickle.load(f2)
#    f2.close()
#    f3 = open("/home/itadmin/NRC_con.pickle", 'rb')
#    Conclassifier = pickle.load(f3)
#    f3.close()
#    f4 = open("/home/itadmin/NRC_ext.pickle", 'rb')
#    Extclassifier = pickle.load(f4)
#    f4.close()
#    f5 = open("/home/itadmin/NRC_neu.pickle", 'rb')
#    Neuclassifier = pickle.load(f5)
#    f5.close()
#    ///////READING THE POSITIVE TEXT FILE//////////////////
#    pos_sent = open("/home/itadmin/positive.txt").read()
#    positive_words = pos_sent.split()
#    //////////READING THE NEGATIVE TEXT FILE ////////////////
#    neg_sent = open("/home/itadmin/negative.txt").read()
#    negative_words = neg_sent.split()
#
#     //////////////////READING THE ANGER TEXT FILE ////////////////
#    anger_sent = open("/home/itadmin/anger.txt").read()
#    anger_words = anger_sent.split()
#
#
#     //////////////////READING THE FEAR TEXT FILE ////////////////
#    fear_sent = open("/home/itadmin/fear.txt").read()
#    fear_words = fear_sent.split()
#
#
#     //////////////////READING THE JOY TEXT FILE ////////////////
#    joy_sent = open("/home/itadmin/joy.txt").read()
#    joy_words = joy_sent.split()
#
#
#     //////////////////READING THE DISGUST TEXT FILE ////////////////
#    disgust_sent = open("/home/itadmin/disgust.txt").read()
#   disgust_words = disgust_sent.split()
#
#
#     //////////////////READING THE ANTICIPATION TEXT FILE ////////////////
#    anticipation_sent = open("/home/itadmin/anticipation.txt").read()
#    anticipation_words = anticipation_sent.split()
#
#
#     //////////////////READING THE SADNESS TEXT FILE ////////////////
#    sadness_sent = open("/home/itadmin/sadness.txt").read()
#    sadness_words = sadness_sent.split()
#
#
#     //////////////////READING THE SURPRISE TEXT FILE ////////////////
#    surprise_sent = open("/home/itadmin/surprise.txt").read()
#    surprise_words = surprise_sent.split()
#
#
#     //////////////////READING THE TRUST TEXT FILE ////////////////
#    trust_sent = open("/home/itadmin/trust.txt").read()
#    trust_words = trust_sent.split()
#
# ////////READING THE TEST DATA ///////////////////
#    profile_file_path = os.path.join(test_data_dir, "profile/profile.csv")
#    print profile_file_path
#    with open(profile_file_path) as csvfile:
#        reader = csv.DictReader(csvfile)
#        for row in reader:
#            uuid = row['userid']
#            positive =    0
#            negative = 0
#            anger = 0
#	    joy = 0
#            anticipation = 0
#            fear = 0
#            trust = 0
#            disgust = 0
#            sadness = 0
#            surprise = 0
#            df = pd.DataFrame()
#
#  / ///////////////// INITIALISING THE VALUE/////////////////////
#            df['positive'] = 0
#            df['negative'] = 0
#            df['anger'] = 0
#            df['joy'] = 0
#            df['anticipation'] = 0
#            df['fear'] = 0
#            df['trust'] = 0
#            df['disgust'] = 0
#            df['sadness'] = 0
#            df['surprise'] = 0
#            test_file = os.path.join(test_data_dir, "text", uuid+".txt")
#            with codecs.open(test_file, "rb", "ISO-8859-1") as fo:
#                try: 
#                    temp = []
#                    for line2 in fo:
#                        stopword = stopwords.words('english')
#                        for word in tknzr.tokenize(line2):
#                    	    temp2 = word.lower()
#                       	    
#                            if temp2 not in stopword:
#                                
#                                temp.append(temp2)
#                                
# ////////////////// CALCULATING THE COUNT OF EVERY EMOTIONS////////////////////
#                            	if temp2 in positive_words:
#			            positive = positive + 1
#                                    
#                                if temp2 in negative_words:
#                                    negative = negative + 1
#                                    
#			        if temp2 in anger_words:
#                                    anger = anger + 1
#                                    
#                                if temp2 in joy_words:
#                                    joy = joy + 1
#    
#                                if temp2 in anticipation_words:
#                                    anticipation = anticipation + 1
#   
#                                if temp2 in fear_words:
#                                    fear = fear + 1
#  
#                                if temp2 in trust_words:
#                                    trust = trust + 1
#
#                                if temp2 in disgust_words:
#                                    disgust = disgust + 1
# 
#                                if temp2 in sadness_words:
#                                    sadness = sadness + 1
#
#                                if temp2 in surprise_words:
#                                    surprise = surprise + 1
#                    df.set_value(0, 'positive', positive)
#                    df.set_value(0, 'negative', negative)
#                    df.set_value(0, 'anger', anger)
#                    df.set_value(0, 'joy', joy)
#                    df.set_value(0, 'anticipation', anticipation)
#                    df.set_value(0, 'fear', fear)
#                    df.set_value(0, 'trust', trust)
#                    df.set_value(0, 'disgust', disgust)
#                    df.set_value(0, 'sadness', sadness)
#                    df.set_value(0, 'surprise', surprise) 
#                    print (df)
#                    neu = str(round(Neuclassifier.predict(df)[0], 2))
#                    ope = str(round(Opeclassifier.predict(df)[0], 2))
#                    con = str(round(Conclassifier.predict(df)[0], 2))
#                    ext = str(round(Extclassifier.predict(df)[0], 2))
#                    agr = str(round(Agrclassifier.predict(df)[0], 2))
#                   
#                    gender = Genderclassifier.classify(extract_features(temp, word_featuresGender))
#                    age = Ageclassifier.classify(extract_features(temp, word_featuresAge))
#                   print("Gender Prediction ", gender)
#		    print("Age Prediction ", age)
 #               except UnicodeDecodeError: 
#		    gender = "0"
    #            output_file = os.path.join(output_dir, uuid+".xml")
   #             with open(output_file, "w") as out_f:
   #                 attrs = {'\n Id': uuid,
  #                           '\n gender' : "male" if gender == "0" else "female",
  #                           '\n age_group' : age,
  #                           '\n extrovert' : ext,
  #                           "\n neurotic" : neu,
  #                           "\n agreeable" : agr,
  #                           "\n conscientious" : con,
 #                            "\n open" : ope
   #                        }
  #                  tree = ElementTree.Element('user', attrs)
  #                  out_f.write(ElementTree.tostring(tree))


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

