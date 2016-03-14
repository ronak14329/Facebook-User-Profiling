import csv
import os
import urllib2
import json
import os
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

def get_page_data(page_id,access_token):  # this function helps to connect Facebookgraphapi url and extract the corresponding page Id information
    api_endpoint = "https://graph.facebook.com/v2.4/" 
    #page_id = "111843568864"   
    fb_graph_url = api_endpoint+page_id+"?fields=id,category,link&access_token="+access_token
    try:
        api_request = urllib2.Request(fb_graph_url)
        api_response = urllib2.urlopen(api_request)
        
        try:
            return json.loads(api_response.read())
        except (ValueError, KeyError, TypeError):
            return "error"

    except IOError, e:
        if hasattr(e, 'code'):
            return "error"
        elif hasattr(e, 'reason'):
            return "error"


def create_user_category_table(path):    #this function helps to create a user and corresponding categories that they liked and store it in database...
    hash1 = {}
    count=0
    flag = 0
    k=0
    max_count = 0
    earlier_id="c6a9a43058c8cc8398ca6e97324c0fae"  # First userid in the Relation.csv file....
   
    with open("/home/madhuri/category.csv",'r') as file:
        hash1["userid"] =earlier_id
        for line in file:
            values = line.strip()
            hash1[values]=0
       
    with open("map2.csv", 'w') as csvfile:
        with open(path,'r') as data:
           c=0
           csv_writer = csv.writer(csvfile)
           for line in data:
              print "started..."              
              values = line.strip().split(",")
              page_id = values[2]
              user_id = values[1]
              
              if(k==0):            
                  if(user_id!="c6a9a43058c8cc8398ca6e97324c0fae"):
                   
                      print "continuing.."
                      c+=1
                      print c                     
                      continue
              
              k+=1              
              if earlier_id != user_id:              
                  print "enter condition"                                            
                  csv_writer = csv.writer(csvfile)
                  if(count==0):
                      print "enter header"
                      csv_writer.writerow(hash1.keys())
                      count+=1
                  csv_writer.writerow(hash1.values())                     
                      
                  for key in hash1.keys():
                      hash1[key]=0
                                            
                  hash1["userid"]=user_id
                  earlier_id=user_id
              access_token = "1532693080358411|l5_waaBWG_MQr3qHsKIDSD_AObY" #token number which is acquired from Facebook Graph API......
               
              pagedata= get_page_data(page_id,access_token)
              if(pagedata != "error"):
                  if 'category' in pagedata.keys():               
                      key = str(pagedata['category'])
                      print key     
                      if key in hash1.keys():
                         hash1[key]+=1
              max_count+=1        
           csv_writer.writerow(hash1.keys())
           csv_writer.writerow(hash1.values())
           csvfile.close()
    return hash1    

hash2 = create_user_category_table("/home/madhuri/Downloads/555/TCSS555/Train/Relation/Relation.csv")              
#for key in hash.keys():
    #print key   

	
