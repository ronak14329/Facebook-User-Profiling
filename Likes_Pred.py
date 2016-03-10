#import pandas as pd
import urllib2
import json
import os
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
#for merging "extracted 'featureset.csv' with 'profile.csv'

#print "90"
#a = pd.read_csv("map.csv")
#b = pd.read_csv("Profile.csv")
#b = b.dropna(axis=1)
#merged = a.merge(b, on='userid')
#merged.to_csv("outputM.csv", index=False)


def get_age_group(age):
    if 18 <= age <= 24.0:
        return "xx-24"
    elif 25 <= age <= 34:
        return "25-34"
    elif 35 <= age <= 49:
        return "35-49"
    elif 50 <= age:
        return "50-xx"
    else:
        return "xx-24"


def get_age_label(age):
    if age=="xx-24" :
        return 1
    elif age=="25-34":
        return 2
    elif age=="35-49":
        return 3
    elif age=="50-xx":
        return 4
    else:
        return 1

def get_images_labels(path,index,training_set):    
     print "getting features and labels..."
     labels = []
     features = [[]]
     count = 0
     profile_f = os.path.join(path)
     with open(profile_f) as fd:
        fd.readline()
        for line in fd:
             feature = []
             #label = []
             if(count==0):
                count+=1
                continue
             if count>training_set:             
                break
             else:
                values = line.strip().split(",")              
                if(index==191):
                    ls=float(values[index])                                   
                        #get the corresponding age bucket..
                    age_bucket = get_age_group(ls)                                
                    l = int(get_age_label(age_bucket))
                    labels.append(l)
                else:
                    l = float(values[index])
                    labels.append(int(l))      
                
                for i in range(1,190):
                    if i!=45:                    
                        feature.append(int(values[i]))                          
                
                features.append(feature)                                           
             count+=1
                                       
     return features,labels,count 

def get_testdata(path,index,training_set):
     print "getting test features and labels..."
    
     labels = []
     features = [[]]
     count = 0
     profile_f = os.path.join(path)
     with open(profile_f) as fd:
        fd.readline()
        for line in fd:
             feature = []
             #label = []
             if count<=training_set:
                count+=1             
                continue
             else:                
                #print "entering else block.."            
                values = line.strip().split(",")
                if(index==191):
                    ls=float(values[index])                                   
                        #get the corresponding age bucket..
                    age_bucket = get_age_group(ls)                                
                    l = int(get_age_label(age_bucket))
                    labels.append(l)
                else:
                    l = float(values[index])
                    labels.append(int(l))       
                for i in range(1,190):
                    if i!=45: #to avoid userids from vector...
                        v=int(values[i])                   
                        feature.append(v)                          
                
                features.append(feature)                                           
             count+=1
                                       
     return features,labels,count



if __name__ == "__main__":
	print "enter path to the 'outputM.csv' file, which is uploaded on Github..."
        path = raw_input("enter path to the 'outputM.csv' file, which is uploaded on Github...\n")
        print path
	traningset = (9501) * 70/100
	#Predictions for age...
	features,labels,count = get_images_labels(path,191,traningset) #index 191 denotes "age coulmn" in the outputM.csv data set....and 192 denotes gender column....
	# outpjutM.csv contains the dataset information which we have extracted from Facebook graph API...
	#test data ...
	t_fet,t_labels,t_count = get_testdata(path,191,traningset)

	#applying random forest classifier.....
	randForest = RandomForestClassifier(n_estimators = 500, n_jobs = 1)
	#print type
	features.remove([])
	randForest.fit(features,labels)
	t_fet.remove([])
	predFeature = randForest.predict(t_fet)
	accuracy = accuracy_score(t_labels,predFeature)
	print ("Predicted model accuracy for Age :"+str(accuracy)) 


	#predictions for gender......
	features,labels,count = get_images_labels(path,192,traningset) #index 191 denotes "age coulmn" in the outputM.csv data set....and 192 denotes gender column....

	# outpjutM.csv contains the dataset information which we have extracted from Facebook graph API...
	#test data ... 
	t_fet,t_labels,t_count = get_testdata(path,192,traningset)

	#applying random forest classifier.....
	randForest = RandomForestClassifier(n_estimators = 500, n_jobs = 1)
	features.remove([])
	randForest.fit(features,labels)
	t_fet.remove([])
	predFeature = randForest.predict(t_fet)
	accuracy = accuracy_score(t_labels,predFeature)
	print ("Predicted model accuracy for Gender :"+str(accuracy)) 

	#RMSEs for Personality traits...

	def rmse(y,x):
	     return np.sqrt(np.mean((y-x)**2))

	#Openness...
	features,labels,count = get_images_labels(path,193,traningset)
	t_fet,t_labels,t_count = get_testdata(path,193,traningset)
	randForest = RandomForestClassifier(n_estimators = 500, n_jobs = 1)
	features.remove([])
	randForest.fit(features,labels)
	t_fet.remove([])
	predFeature = randForest.predict(t_fet)
	accuracy = accuracy_score(t_labels,predFeature)
	print ("RMSE error for Openenss:" + str(rmse(predFeature,t_labels)))

	features,labels,count = get_images_labels(path,194,traningset)
	t_fet,t_labels,t_count = get_testdata(path,194,traningset)
	randForest = RandomForestClassifier(n_estimators = 500, n_jobs = 1)
	features.remove([])
	randForest.fit(features,labels)
	t_fet.remove([])
	predFeature = randForest.predict(t_fet)
	accuracy = accuracy_score(t_labels,predFeature)
	print ("RMSE error for conscientiousness :" + str(rmse(predFeature,t_labels)))


	features,labels,count = get_images_labels(path,195,traningset)
	t_fet,t_labels,t_count = get_testdata(path,195,traningset)
	randForest = RandomForestClassifier(n_estimators = 500, n_jobs = 1)
	features.remove([])
	randForest.fit(features,labels)
	t_fet.remove([])
	predFeature = randForest.predict(t_fet)
	accuracy = accuracy_score(t_labels,predFeature)
	print ("RMSE error for extraversion :" + str(rmse(predFeature,t_labels)))

	features,labels,count = get_images_labels(path,196,traningset)
	t_fet,t_labels,t_count = get_testdata(path,196,traningset)
	randForest = RandomForestClassifier(n_estimators = 500, n_jobs = 1)
	features.remove([])
	randForest.fit(features,labels)
	t_fet.remove([])
	predFeature = randForest.predict(t_fet)
	accuracy = accuracy_score(t_labels,predFeature)
	print ("RMSE error for agreeableness :" + str(rmse(predFeature,t_labels)))

	features,labels,count = get_images_labels(path,197,traningset)
	t_fet,t_labels,t_count = get_testdata(path,197,traningset)
	randForest = RandomForestClassifier(n_estimators = 500, n_jobs = 1)
	features.remove([])
	randForest.fit(features,labels)
	t_fet.remove([])
	predFeature = randForest.predict(t_fet)
	accuracy = accuracy_score(t_labels,predFeature)
	print ("RMSE error for neuroticism :" + str(rmse(predFeature,t_labels)))


	#K nearest neighbor algorithm....

	clf = neighbors.KNeighborsClassifier()
	#clf.fit(features,labels)
	#Z = clf.predict(t_fet)
	#print "prediction done..."
	#accuracy=accuracy_score(t_labels,Z)
	#print ("Predicted model accuracy: "+ str(accuracy))     

	#Gaussian NaiveBayes....
	gnb = GaussianNB()
	#y_pred = gnb.fit(features,labels).predict(t_fet)
	#accuracy = accuracy_score(t_labels,y_pred)
	#print("Predicted model accuracy:"+str(accuracy))

	#applying Svm..
	 

	clf = svm.SVC()
	#clf.fit(features,labels)
	#y_pred = clf.predict(t_fet)
	#accuracy = accuracy_score(t_labels,y_pred)
	#print ("Predicted model accuracy: "+ str(accuracy)) 

	#RMSE calculations for Personality traits....
	#randForest.fit(features,labels)
	#t_fet.remove([])
	#predFeature = randForest.predict(t_fet)





