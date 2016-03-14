============================================================ 
3 To Run Tcss5551 file, please fallow below steps 
4 ============================================================ 
5 
 
6 1.Software Installations: 
7 
 
8      Install python 2.7 version 
9           on ubuntu/linux: on terminal type: sudo apt-get install python 
10           
11      Install scikit learn package 
12           on ubuntu/linux: on terminal type: pip install -U scikit-learn 
13 
 
14      Install numpy package 
15           on ubuntu/linux: on terminal type: sudo apt-get install python-numpy 
16 
 
17      Install scipy package 
18           on ubuntu/linux: on terminal type: sudo apt-get install python-scipy 
19 
 
20      Install Opencv:  
21           on ubuntu/linux: sudo apt-get install python-opencv 
22 
 
23      Install Tensorflow: URL : https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html 
24               
25             
26 
 
27 2. Inside the code in main function, change the path location of "trainingdir" to the directry of training data set on your local system. 
28 
 
29 3.Run the program using terminal window: ./tcss555_images -i /data/public-test-data -o ~/outputDir 
30 
 
31 4.The program will run and produce output XML files. 
32 
 
33 ========================================================================== 
34 Code Description 
35 ==========================================================================  
36 
 
37 The code is implemented in python using OpenCv Face Recognizer and TensorFlow deep neural network algorithms. 
38 
 
39 In OpenCV, I have used HaarCascade classifer and LBPH face recognizer techniques. 
40 Intially I convert the image into gray scale and then crop the face using harr cascades. 
41 
 
42 Resized the image into specific size. So, that all the images would be in same size for training. 
43 
 
44 Inside OpenCv there are existing functions like 'train', which takes inpput as set of images and labels. 
45 
 
46 There is also one builtin 'predict' method on the top of trained object. 
47 
 
48 'predict' method accepts set of testing images and returns a list of corresponding label predictions. 
49 
 
50 We have used scikit-learn accuracy score measure function, to measure the accuracy of above predicted labels Vs Actual labels. 
51 
 
52 Coming to TensorFlow, 
53 
 
54 I converted the Image into gray scale 
55 
 
56 I converted the image into multi dimensional numeric array, since tensor flow works on images as a numerical array values 
57 
 
58 I constructed a graph network, where each node represents a weight update value and edges reresents the multi dimensional numeric value array.  
59 
 
60 This can be done in oneline using tensor flow (tf.session()) 
61 
 
62 I have used Gradient Decent weight update rule to minimize 'cross-entropy' value 
63 
 
64 The internal working process is similar to back propogation algorithm 
65 
 
66 
 
67 Ref : url: https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html#setup 
68 
 
69 
 
70 
 
71 
 
72 ============================================================================================== 
73 Functions Description 
74 ============================================================================================== 
75 
 
76 def predict_gender_labels(path,testpath,recognizer): For each image in the testset, the recognizer returns predicted label.  
77 Finally function returns a list of all predicted labels 
78 
 
79 def public_test_data(path):Prepare the test data. Basically we exctraced each image and did preprocessing steps (crop, resize,grayscale, etc) 
80 
 
81 
 
82 def test_images_and_labels(path,index,training_set): 
83 
 
84 cascadePath = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') : Opencv already contains pre-trained classifiers for face, eyes, smile etc. 
85 
 
86 So loading the XML that contains these classifiers. 
87 
 
88 faceCascade = cv2.CascadeClassifier(cascadePath)  
89 
 
90    
91 faces=faceCascade.detectMultiScale(image_pic): If faces are found, it returns the position of detected faces as Rect(x,y,w,h) 
92      
93 recognizer = cv2.createLBPHFaceRecognizer() : For face recognition we will use the the LBPH Face Recognizer  
94 
 
95 recognizer.train(t_images, np.array(t_labels)) : In opencv there is 'train' method that we can use to train our model 
96 
 
97 nbr_predicted,conf = recognizer.predict(image): In opencv there is 'predict' method that we can call  
98 
 
99 for predictions. It returns a predicted label and confidence level 
100 
 
101 
 
102 def testdata(directry): Reads all the test images and convert each image into a numpy array and store it as a list of lists for testing. 
103 
 
104 
 
105 def flatten(vec): It helps fatten the image vector. 
106 
 
107 
 
108 def imagevector(directry): It helps to create a set of numpy arrays for training data. 
109 
 
110 
 
111 def train_age(data_gen, test_gen): Train the model to predict age. Basically, training the neural network based on label age. In tensorflow this 
112   
113 can be done in few steps. Each step in the code contains comments, so that it helps to understand better how it is working. 
114 
 
115 
 
116 def train_and_test(data_gen, test_gen): Train the model to predict gender label. Same code as above, but the prediction label is different. 
117 
 
118 
 
119 
 
120 
 
121 
 
