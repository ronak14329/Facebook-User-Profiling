#!/usr/bin/python

# Import the required modules
import cv2, os
import argparse
import numpy as np
import csv
import Image
from skimage import io
from sklearn.metrics import accuracy_score
import tensorflow as tf
from xml.etree import ElementTree
from skimage.color import rgb2gray
from skimage import transform
import numpy
import math, sys


# For face detection we will use the Haar Cascade provided by OpenCV.
#cascadePath = "/home/madhuri/Downloads/face_recognizer/haarcascade_frontalface_default.xml"
#faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will use the the LBPH Face Recognizer 
#recognizer = cv2.createLBPHFaceRecognizer()
#recognizer = cv2.createEigenFaceRecognizer()



#define age buckets.....

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

#writing output files.....
def write_out(out_dir, predicted):   
    
    for prediction in predicted:
        id,gender,age = prediction
        print gender        
        out_file = os.path.join(out_dir, id+".xml")
        if gender==1:
            gender="female"
        else:
            gender = "male"
        print gender
        with open(out_file, "w") as fd:
            attrs = { "userId" : id,
                      "gender" : gender,                       
                      "age_group" : get_age_group(age),
		      'extrovert': str(3.4869),
                      'neurotic': str(2.7324),
                      'agreeable': str(3.5839),
                      'conscientious': str(3.4456),
                      'open': str(3.9087)
                    }
            tree = ElementTree.Element("user", attrs)
            fd.write(ElementTree.tostring(tree))

#To get the total number of Rows in training data....

def getRows(path):
    count=0
    fullpath = os.path.join(path,"Train/Profile", "Profile.csv")
    with open(fullpath) as fd:
        for line in fd:
            count = count+1
    return count


def train_d():
    return train_m;
         

def get_images_labels(path,index,training_set):    
     print "getting images and labels..."
     labels = []
     images = []
     count = 0
     profile_f = os.path.join(path,"Train/Profile", "Profile.csv")
     with open(profile_f) as fd:
        fd.readline()
        for line in fd:
             if count>training_set:             
                break
             else:
                if count==0:
                    count+=1
                    continue
                
                else:
                    print "entering else block.."
                    values = line.strip().split(",")
                    userid = values[1]
                    if(userid=="userid"):
                        continue
                    elif(index==3):
                        label=float(values[3])
                        label_s=int(label)
                    else:
                        label=float(values[2])
                        l_s=int(label)                 
                        #get the corresponding age bucket..
                        age_bucket = get_age_group(l_s)                                
                        label_s = int(get_age_label(age_bucket))
 
                    image_file = os.path.join(path,"Train/Image", userid+".jpg")  
                    
                    image_pics =Image.open(image_file).convert('L')                                       
                    image_pic = np.array(image_pics, 'uint8')        
                                     
                    #apply faceCascade detections....
                    faces=faceCascade.detectMultiScale(image_pic)
                    print "did the cascades....."
                    for (x,y,w,h) in faces:
                        #cropping the image...
                        crop_img = image_pic[y: y + h, x: x + w]                      
                        #resized_image = cv2.resize(crop_img,(100,100))              
                        #append the resized image....
                        images.append(crop_img)
                        labels.append(label_s)
                        cv2.imshow("Add", crop_img)
                                       
                    
                    count+=1
                                       
     return images,labels 



#recognizer.train(images,labels)
def test_images_and_labels(path,index,training_set):
    
     labels = []
     images = []
     count=0
     tc=0
     profile_f = os.path.join(path,"Train/Profile","Profile.csv")
     with open(profile_f) as fd:
        fd.readline()
        for line in fd:
            if count<training_set:
                count+=1
                continue                       
            else:            
                values = line.strip().split(",")
                userid = values[1]
                if(index==3):
                    label=float(values[3])
                    label_s=int(label)
                else:
                    label=float(values[2])
                    l_s=int(label)                 
                        #get the corresponding age bucket..
                    age_bucket = get_age_group(l_s)                                
                    label_s = int(get_age_label(age_bucket))
 
                image_file = os.path.join(path,"Train/Image",userid+".jpg")  
                    
                image_pics =Image.open(image_file).convert('L')                                       
                image_pic = np.array(image_pics, 'uint8')   
                #Apply face cascade methods....              
                faces=faceCascade.detectMultiScale(image_pic)
                for (x,y,w,h) in faces:                    
                    crop_img = image_pic[y: y + h, x: x + w]
                    #resized_image = cv2.resize(crop_img,(100,100))                    
                    images.append(crop_img)                
                    
                    labels.append(label_s)
                    cv2.imshow("Adding faces to testing set...", image_pic[y: y + h, x: x + w])
                      
                    #cv2.waitKey(500)
                tc+=1     
                count+=1                                
     return images, labels  


def public_test_data(path):
    test_images=[]
    pathfile = os.path.join(path,"Public Test/Profile","Profile.csv")
    with open(pathfile) as fd:
        for line in fd:
            values = line.strip().split(",")
            userid = values[1]
            if(userid=="userid"):
                continue
            #print userid
            image=os.path.join(path,"Public Test/Image",userid+".jpg")
            image_pics = Image.open(image).convert('L')
            image_pic = np.array(image_pics,'uint8')
            faces=faceCascade.detectMultiScale(image_pic)
            for (x,y,w,h) in faces:                    
                    crop_img = image_pic[y: y + h, x: x + w]
                    #resized_image = cv2.resize(crop_img,(100,100))                    
                    test_images.append(crop_img) 
    return test_images

def predict_gender_labels(path,recognizer):    
    t_images = public_test_data(path)
    predictor=predict_gender(path,recognizer)
    predicted = []
    for image in t_images:
        nbr_predicted,conf= predictor.predict(image)      
        predicted.append(nbr_predicted)
    return predicted

def predict_gender(training_path,recognizer):
    nrow=getRows(training_path)       
    training_set = nrow*75/100
    #print training_set
    test_set=nrow-training_set
    
    t_images,t_labels = get_images_labels(training_path,3,training_set)    
    cv2.destroyAllWindows()

    # Perform the tranining
    recognizer.train(t_images, np.array(t_labels))
    #test images and labels....    
    t_images,t_labels = test_images_and_labels(training_path,3,training_set)    
    c=0     
    predicted = []
    for image in t_images:
        nbr_predicted,conf = recognizer.predict(image)     
         
        if(nbr_predicted==t_labels[c]):
            print "{} is Correctly Recognized with confidence {}".format(t_labels[c], conf)
            c+=1
               
        predicted.append(nbr_predicted)
        
    print 'Accuracy score is: {0}'.format(accuracy_score(t_labels,predicted))        

    return recognizer


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
                        help='Full path to input test directory containing profile and text dir')
                        
    parser.add_argument('-o', "--output_dir",
                        type=str,
                        default='output',
                        help='The path to output directory')
                    

    args = parser.parse_args()
    return args



def imagevector(directry):
    
    
    #get training data as userid,gender,numpy array....
   
    profile_f = os.path.join(directry, "Train/Profile", "Profile.csv")
    
    with open(profile_f) as fd:
        fd.readline()
        for line in fd:
            values = line.strip().split(",")
            age = float(values[2])
            sex = float(values[3])
            userid = values[1]
            if(userid=="userid"):
               continue
            image_file = os.path.join(directry, "Train/Image", userid+".jpg")
            numpy_array = io.imread(image_file, as_grey=True)            
            yield(userid, sex, numpy_array, age)

def flatten(vec):   
    zz = vec.flatten()
    size = 50 * 50
    zero_v = numpy.zeros(50 * 50)
    
    for idx in range(min(len(zz), size)):
        zero_v[idx] =  zz[idx]        
    return zero_v

def testdata(directry):
    profile_f = os.path.join(directry, "Public Test/Profile", "Profile.csv")
    
    with open(profile_f) as fd:
        fd.readline()
        for line in fd:
            values = line.strip().split(",")
            userid = values[1]
            image_file = os.path.join(directry,  "Public Test/Image", userid+".jpg")
            numpy_array = io.imread(image_file, as_grey=True)
            
            yield(userid, numpy_array)    


def train_age(data_gen, test_gen):    
    dim = 50 * 50
    xx = tf.placeholder(tf.float32, [None, dim])
    WW = tf.Variable(tf.zeros([dim, 1]))
    bb = tf.Variable(tf.zeros([1]))
    yy = tf.add(tf.matmul(xx, WW), bb)
    yy_a = tf.placeholder(tf.float32, [None, 1]) 
    cost = tf.pow(yy_a - yy, 2)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    train_y = []
    train_x = []
    for idx, data in enumerate(data_gen):
        if idx > 100:
            break
        _, sex, n_arr, age = data
        age = float(age)
        train_x.append(flatten(n_arr))
        train_y.append(numpy.array([age]))

    print("Trained model....")
 
    sess.run(train_step, feed_dict={xx: train_x, yy_a: train_y})
    wt = sess.run(WW)
    bias = sess.run(bb)   
    
    correct_prediction = tf.equal(tf.argmax(yy_a,1), tf.argmax(yy,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print sess.run(accuracy, feed_dict={xx: train_x, yy_a: train_y})
    #print 'Accuracy score is: {0}'.format(accuracy_score(train_x,train_y))        

     
    test_x = []
    for _, numpy_arr in test_gen:
        test_x.append(flatten(numpy_arr))

    yy_p = sess.run(yy, feed_dict={xx: test_x})
    age = []
    for item in  yy_p:
        age.append(item[0]/ 100000.0)
    return age



if __name__ == "__main__":
    args = parse_args()
    test_dir = args.test_dir
    training_dir= test_dir
    uids = []
    for id, _ in testdata(test_dir):
        uids.append(id)     
 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
     
    cascadePath = "/home/madhuri/Downloads/face_recognizer/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # For face recognition we will use the the LBPH Face Recognizer 
    recognizer = cv2.createLBPHFaceRecognizer()  
    gender = predict_gender_labels(training_dir,recognizer)
    
    age = train_age(imagevector(training_dir), testdata(test_dir))
    u_len = len(uids)
    g_len = len(gender)
    dif = u_len-g_len
    while(dif>0):
        gender.append(1)
        dif=dif-1
    write_out(args.output_dir, zip(uids,gender,age))


