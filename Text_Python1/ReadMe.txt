============================================================
To Run Text.py file, please fallow below steps
============================================================

1.Software Installations:

     Install python 2.7 version
          on ubuntu/linux: on terminal type: sudo apt-get install python
         
     Install scikit learn package
          on ubuntu/linux: on terminal type: pip install -U scikit-learn

     Install numpy package
          on ubuntu/linux: on terminal type: sudo apt-get install python-numpy

   Install nltk package
          on ubuntu/linux: on terminal type: sudo apt-get install python-nltk
          
2. Inside the code in main function, change the path location of "trainingdir" to the directry of training data set on your local system.

3.Run the program using terminal window: Text.py -i /data/public-test-data -o ~/outputDir

4.The program will run and produce output XML files.          

==========================================================================
Code Description
========================================================================== 
In this for the Age and Gender prediction we have used the Naive Bayes.
I have used the Tweet Tokenizer function to seperate the words
For removing the stop words I have used the stopwordcorpus.
Tolower function to convert the words into the lower case.
After this preprocessing we have generated the Pickle file which contain the status along with their respective labels
For AgeTweets.pickle file which contain the tweets along with its age label.
For GenderTweets.pickle file which contain the tweets along with its gender label.
We have also generated the Gender_classifier2.pickle and Age_classifier2.pickle for the classification.
All this pickle file can be called in the main function using the function pickle.load().

For Personality Prediction 
Initially I have used the LIWC features for predicting the Personality traits using all its features.
But it doesn't give results above the baseline.
So i have tried using the positive and negative sentiments approach.
In this I have read each words from the status of the test data and compare the words in the positive or negative text file.
If the word found in the positive file it increment the positive count by 1 and if the word is found in the negative file it increment the negative count by 1 and if the word not found in both the files then we increment the neutral count by 1.
Using this counts we can predict the presonality traits.

For better Accuracy 
I have tried Using the NRC lexicon Emotion features which contain 10 emotions along with the positive and negative sentiments.it works same as above. 

==============================================================================================
Functions Description
==============================================================================================
def get_tweetsword(tweets):This function is used to get the words from the tweets and store it a list

def get_word_features(wordlist):This function will count the frequency of the words.

def extract_features(document, word_features): This function will extract the features.

def predict_and_write(Genderclassifier, Ageclassifier, test_data_dir, output_dir, Age_word_features, Gender_word_features):
This is the function used to predict the gender,age and Personlaity traits.

def run(test_dir, output_dir): This function is used to get all the agetweets and GenderTweets pickle file and used them in the program

def parse_args():This function is used to get the input and output directory address from the users.

TweetTokenizer():- This function is used to seperate the words.

Stopword:- This function is used to remove all the stop words.








To run this we need pickle file which I have already uploaded . Since the size of Pickle file is larger than 25MB so have compressed it.
Please unzip the pickle file before using.
