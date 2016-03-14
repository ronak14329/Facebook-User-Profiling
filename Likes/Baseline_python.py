import csv
#import math
from math import *
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
import os
from xml.etree import ElementTree
#import nltk
#from nltk.tokenize import word_tokenize
#from sklearn import datasets, svm, metrics
#from hyperopt import hp


def write_out(out_dir,predictedValues,uids):
    """
    """
    for id in uids:        
        out_file = os.path.join(out_dir, id+".xml")
        gender = predictedValues[0]
        age = predictedValues[1]
        with open(out_file, "w") as fd:
            attrs = { "userId" : id,
                      "gender" : "female" if gender else "male",
                      "age_group" : get_age_bucket(age),
		      'extrovert': str(3.4869),
                      'neurotic': str(2.7324),
                      'agreeable': str(3.5839),
                      'conscientious': str(3.4456),
                      'open': str(3.9087)
                    }
            tree = ElementTree.Element("user", attrs)
            fd.write(ElementTree.tostring(tree))


def get_age_bucket(age):
   if age==1:
       return "xx-24"
   else:
       return "25-34"

def parse_args():
    parser = argparse.ArgumentParser(description="""Script takes full input path to
                         test directory, output directory and training directory""")

    parser.add_argument('-d',
                        "--training_dir",
                        default='',
                        type=str, 
                        help='Full path to input trainig directory')

    parser.add_argument('-i',
                        "--test_dir",
                        type=str, 
                        required=True,
                        help='Full path to input test directory containing profile,Image,Text dir')
                        
    parser.add_argument('-o', "--output_dir",
                        type=str,
                        default='output',
                        help='The path to output directory')
                    

    args = parser.parse_args()
    return args


def countLines(read):
    count=0
    for i in read:        
        count+=1
    return count

def readfile(path):
     fp=open(path)
     read=csv.reader(fp)
     return read
    
def baseline1_prediction():
    read=readfile("/home/madhuri/Downloads/555/TCSS555/Train/Profile/Profile.csv")
    count=0
    for line in read:        
        print(line)
              
read=readfile("/home/madhuri/Downloads/555/TCSS555/Train/Profile/Profile.csv")    
length=int(countLines(read))
print(length)
traing_set=int(length*70/100)
testing_set = length-traing_set
print(traing_set)
print(testing_set)

def baseline_prediction_gender(path,traing_set):   
    count=0
    female=0
    male=0
    targetval=0   
    read=readfile(path) 
    for line in read:        
        if count>traing_set:
            break    
        else:
            if count==0:
                count+=1
                continue
            else:
                
                if(line[3]=='1'):                   
                    female+=1                
                else:
                    male+=1
                count+=1
    
    if(female>male):        
        targetval=1
    
    return targetval


def get_age_group(age):
    if 18 <= age <= 24.0  :
        return 1
    elif 25 <= age <= 34:
        return 2
    elif 35 <= age <= 49:
        return 3
    elif 50 <= age:
        return 4
    else:
        return 1

def age_baseline(path,target_set):
    ages=[]
    count=0
    read = readfile(path)
    for line in read:
        if count>target_set:
            break
        else:
            if count==0:
                count+=1
                continue
            else:
                age = float(line[2])
                age_bucket = get_age_group(int(age))
                ages.append(age_bucket)
    return 1

def personality(path,index):
    count=0
    sum=0.0
    read=readfile(path)
    personolities = []
    for line in read:        
        if count>traing_set:
            break
        else:
            if count==0:
                count+=1
            else:                
                personolities.append(float(line[index]))
                count+=1                
    personolity_arry = np.array(float(personolities))
    return np.mean(personolity_arry)
    
actual_gender = []                
def testSet(read,limit,path):
    count=0
    testcount=0
    i=0
    list=[]
    g_count=0
    a_count=0
    actual_gender = []
    predicted_gender = []
    actual_age = []
    predict_age = []
    for line in read:       
        if(count<limit):
            count+=1
            continue
        else:            
            list.append(line)
            gender = float(line[3])
            gender = int(gender)
            actual_gender.append(gender)
            p_gender = int(baseline_prediction_gender(path,limit))
            if(gender == p_gender):
                g_count+=1 
            predicted_gender.append(p_gender)
            age=float(line[2])
            age=int(age)
            age_bucket=get_age_group(age)
            actual_age.append(age_bucket)
            if(age==age_bucket):
                a_count+=1
            predict_age.append(age_baseline(path,limit))
            testcount+=1     
  
    
    print 'Baseline Gender Accuracy score is: {0}'.format(accuracy_score(actual_gender,predicted_gender))
    print 'Baseline Age Accuracy score is: {0}'.format(accuracy_score(actual_gender,predicted_gender))
    return list
 
 

def testdata_personality(read,index):
    accurate_count=0
    total_instances=countLines(read)
    val=personality(index)
    low = floor(val)
    high=round(val)
    print("absolute value is",low)
    print("absolute value is",high)
    for i in range(0,total_instances):
        attributes=read[i][4]
               
        if(float(attributes)>=low and float(attributes)<=high):
            accurate_count+=1
    
    return (accuracy_count/total_instances)*100        


if __name__ == "__main__":
    args = parse_args()
    test_dir = args.test_dir
    test_path= os.path.join(test_dir,"Public Test/Profile","Profile.csv")
    read = readfile(test_path)
    uids = []
    for line in read:
        id = line[1]
        uids.append(id)     
 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    profile_f = os.path.join(test_dir, "Train/Profile", "Profile.csv")
    read=readfile(profile_f)
    length=int(countLines(read))
    print(length)
    traing_set=int(length*70/100)
    testing_set = length-traing_set
    print(traing_set)
    print(testing_set)
    testdata=testSet(readfile(profile_f),traing_set,profile_f)
    p_gender = int(baseline_prediction_gender(profile_f,traing_set))
    p_age=age_baseline(profile_f,traing_set)
    predicted =[]
    predicted.append(p_gender)
    predicted.append(p_age)
    write_out(args.output_dir,predicted,uids)   
    
    
    

