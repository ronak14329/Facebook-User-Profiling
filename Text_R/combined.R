###########################################################################
# Author: Radha Madhavi Malireddy
#
# Description: R Models for automatic recognition of the age, gender, and 
# 	personality of Facebook users using the features extracted from 
#	  Status Updates Text.
#
# Input Arguments: This script expects the following arguments
# 	-i <TestDir> -o <ResultsDir>
# 
# Assumptions: 
# 	1. This script assumes that the Training data is placed in 
# 	"/data/training" directory in the same layout as in the shared VM.
# 	2. The R packages required for this Script are already installed on the 
# 	target machine. 
###########################################################################

# Switch to use either Random forests or Linear regression for predicting personality Traits
useRandomForests = FALSE; 

options(mc.cores = 1)

# 0. Parse the command line arguments to get the path to the test data and results directory
args = commandArgs(TRUE)

TestDir = args[2]
ResultsDir = args[4]
if(args[3] == "-i") {
  TestDir = args[4] 
  ResultsDir = args[2]  
}

# assume the Training data directory
TrainingDir = "/data/training"

#print(TestDir)
#print(ResultsDir)
#print(TrainingDir)

# Install packages and add libraries
#install.packages("tm")
#install.packages("SnowballC")
#install.packages("readr")
#install.packages("e1071")
#install.packages("gmodels")
#install.packages("randomForest")
library(tm)
library(SnowballC)
library(readr)
library(e1071)
library(gmodels)
library(caret)
library(class)
library(hydroGOF)
library(randomForest)

# 1. Read the Profile data for training and test data
profile.raw_train = read.csv(paste(TrainingDir, "profile/profile.csv", sep="/"), stringsAsFactors=FALSE)
profile.raw_ptest = read.csv(paste(TestDir, "profile/profile.csv", sep="/"), stringsAsFactors=FALSE)

#2. Prepare the data

# Based on the structure information profile.raw_train$gender is read as numeric variable. We need to format
# it as a factor for fitting the training data. 
profile.raw_train$genderclass = as.factor(ifelse(profile.raw_train$gender == 1, "female", "male"))

# Based on the structure information profile.raw_train$age is read as numeric variable. We need to format
# as a multi-level factor variable for fitting the training data. 
# note: We can also use "findInterval" for bucketizing the age. 
profile.raw_train$ageclass = cut(profile.raw_train$age, 
    breaks = c(-Inf, 25, 35, 50, Inf), 
    labels = c("xx-24", "25-34", "35-49", "50-xx"), 
    right = FALSE)

# Read the status updates text from the <user-id>.txt files and merge it into
# column vectors for training and test data
for(i in 1:length(profile.raw_train$userid))
{
  profile.raw_train$statusupdates[i] = read_file(paste(paste(TrainingDir, "text", sep="/"), paste(profile.raw_train$userid[i], "txt", sep = "."), sep="/"))
}

for(i in 1:length(profile.raw_ptest$userid))
{
  profile.raw_ptest$statusupdates[i] = read_file(paste(paste(TestDir, "text", sep="/"), paste(profile.raw_ptest$userid[i], "txt", sep = "."), sep="/"))
}

# Status updates are strings of text composed of words, spaces, numbers, and punctuation.
# We need to remove numbers, punctuation, handle uninterresting words such as and, but and or. 
# all of the text mining processing can be done using "tm" package in R. 
# cleanCorpus function is used for doing the text preprocessing

cleanCorpus <- function(corpus) {
  #temp = tm_map(corpus, content_transformer(function(x) iconv(enc2utf8(x), sub = "byte"))) # new step
  temp = tm_map(corpus, content_transformer(function(x) iconv(x, "ASCII", "UTF-8", sub=""))) # new step
  corpus_clean = tm_map(temp, content_transformer(tolower), lazy = TRUE)
  corpus_clean = tm_map(corpus_clean, removePunctuation,lazy = TRUE)
  corpus_clean = tm_map(corpus_clean, removeNumbers,lazy = TRUE)
  corpus_clean = tm_map(corpus_clean, removeWords, stopwords("english"),lazy = TRUE)
  corpus_clean = tm_map(corpus_clean, stripWhitespace,lazy = TRUE)
  corpus_clean = tm_map(corpus_clean, stemDocument,language = "english", lazy = TRUE)
  corpus_clean = tm_map(corpus_clean, PlainTextDocument,lazy = TRUE) 
  return(corpus_clean)
}

#clean up step to unclutter the status updates text of the training observations
statusupdates_corpus = Corpus(VectorSource(profile.raw_train$statusupdates))
corpus_clean = cleanCorpus(statusupdates_corpus)

#clean up step to unclutter the status updates text of the test data
statusupdates_corpus_ptest = Corpus(VectorSource(profile.raw_ptest$statusupdates))
corpus_clean_ptest = cleanCorpus(statusupdates_corpus_ptest)


# Tokenization: creates Sparse Matrix with unigram term frequencies
su_dtm_train = DocumentTermMatrix(corpus_clean)
su_dict_train = findFreqTerms(su_dtm_train, 40) 
su_train = DocumentTermMatrix(corpus_clean, list(dictionary=su_dict_train))
su_test = DocumentTermMatrix(corpus_clean_ptest, list(dictionary=su_dict_train))

# Helper function for creating indicator features
convert_counts = function (x) {
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return (x)
}

# Create indicator features
su_train = apply(su_train, MARGIN = 2, convert_counts)
su_test = apply(su_test, MARGIN = 2, convert_counts)

#
# Gender Model using unigram indicator featuers
#
nb_classifier_genderclass_laplace = naiveBayes(su_train, profile.raw_train$genderclass, laplace = 1)
su_test_pred_genderclass_laplace = predict(nb_classifier_genderclass_laplace, su_test)
#CrossTable(su_test_pred_genderclass_laplace, profile.raw_test$genderclass, prop.chisq = FALSE, prop.t=FALSE, dnn = c('predicted', 'actual'))
#confusionMatrix(su_test_pred_genderclass_laplace, profile.raw_test$genderclass)

#
# Age Model using unigram indicator features
#
nb_classifier_ageclass_laplace = naiveBayes(su_train, profile.raw_train$ageclass, laplace = 1)
su_test_pred_ageclass_laplace = predict(nb_classifier_ageclass_laplace, su_test)
#CrossTable(su_test_pred_ageclass_laplace, profile.raw_test$ageclass, prop.chisq = FALSE, prop.t=FALSE, dnn = c('predicted', 'actual'))
#confusionMatrix(su_test_pred_ageclass_laplace, profile.raw_test$ageclass)

##
# Personality Traits using LIWC features
##

# read LIWC.csv for Training and Test Data
# Merge the frames into profile.csv data frame
# Merge is necessary as profile.csv and LIWC.csv has userid column in different order. 
liwc.train = read.csv(paste(TrainingDir, "LIWC.csv", sep="/"), stringsAsFactors=FALSE)
liwc.train$userid = liwc.train$userId
liwc.train = liwc.train[,-1]  # remove userId column and retain "userid" column
liwc.train = merge(liwc.train, profile.raw_train, by="userid")
liwc.vs = read.csv(paste(TestDir, "LIWC.csv", sep="/"), stringsAsFactors=FALSE)
liwc.vs$userid = liwc.vs$userId
liwc.vs = liwc.vs[,-1]  # remove userId column and retain "userid" column
liwc.vs = merge(liwc.vs, profile.raw_ptest, by="userid")

#
# Note: In all of the following models based on LIWC, Features present in 
# First and Second column are not used as they are same for all users or they are
# not useful for training the learner. 
#

#
# Openess Model
#
liwc.ope.train = liwc.train[,3:83]
liwc.ope.train$ope = liwc.train$ope
liwc.ope.vs = liwc.vs[,3:83]
liwc.ope.vs$ope = liwc.vs$ope

rf.liwc.ope = randomForest(ope ~ ., data = liwc.ope.train, mtry=9, importance=TRUE , ntree=100)
rf.liwc.ope.pred = predict(rf.liwc.ope, liwc.ope.vs)

#
# Conscientiousness Model
#
liwc.con.train = liwc.train[,3:83]
liwc.con.train$con = liwc.train$con
liwc.con.vs = liwc.vs[,3:83]
liwc.con.vs$con = liwc.vs$con

rf.liwc.con = randomForest(con ~ ., data = liwc.con.train, mtry=9, importance=TRUE , ntree=100)
rf.liwc.con.pred = predict(rf.liwc.con, liwc.con.vs)

#
# Extroversion  Model
#
liwc.ext.train = liwc.train[,3:83]
liwc.ext.train$ext = liwc.train$ext
liwc.ext.vs = liwc.vs[,3:83]
liwc.ext.vs$ext = liwc.vs$ext

rf.liwc.ext = randomForest(ext ~ ., data = liwc.ext.train, mtry=9, importance=TRUE , ntree=100)
rf.liwc.ext.pred = predict(rf.liwc.ext, liwc.ext.vs)

#
# Agreeableness Model
#
liwc.agr.train = liwc.train[,3:83]
liwc.agr.train$agr = liwc.train$agr
liwc.agr.vs = liwc.vs[,3:83]
liwc.agr.vs$agr = liwc.vs$agr

rf.liwc.agr = randomForest(agr ~ ., data = liwc.agr.train, mtry=9, importance=TRUE , ntree=100)
rf.liwc.agr.pred = predict(rf.liwc.agr, liwc.agr.vs)

#
# Emotional Stability  Model
#
liwc.neu.train = liwc.train[,3:83]
liwc.neu.train$neu = liwc.train$neu
liwc.neu.vs = liwc.vs[,3:83]
liwc.neu.vs$neu = liwc.vs$neu

rf.liwc.neu = randomForest(neu ~ ., data = liwc.neu.train, mtry=9, importance=TRUE , ntree=100)
rf.liwc.neu.pred = predict(rf.liwc.neu, liwc.neu.vs)


#
# Combine the results from different models
#

# Combine the prediction results for all Big 5 Personality Traits into single data frame
rf.liwc.combined = data.frame(userId = liwc.vs$userid,
                              extrovert = rf.liwc.ext.pred,
                              neurotic = rf.liwc.neu.pred, 
                              agreeable = rf.liwc.agr.pred,
                              conscientious = rf.liwc.con.pred, 
                              open = rf.liwc.ope.pred,  stringsAsFactors = FALSE)


# Combine the prediction results for Age and Gender into single data frame							  
nb.agegender.combined = data.frame(userId = profile.raw_ptest$userid, 
                                   age_group = as.character(su_test_pred_ageclass_laplace),
                                   gender = as.character(su_test_pred_genderclass_laplace),
                                   stringsAsFactors = FALSE)
								   
#merge the results
results = merge(rf.liwc.combined, nb.agegender.combined , by="userId")

#
# Write the prediction results into <ResultsDir> as per description from
# ProjectDescription.pdf uploaded in the canvas
#
for (i in 1:length(results$userId)) {
  outFileName = paste(ResultsDir, paste(results$userId[i], "xml", sep = "."), sep="/")
  outFileCon = file(outFileName, "w")
    
  cat(sprintf('<user \n'),file= outFileCon)    
  cat(sprintf('userId="%s"\n',results$userId[i]),file= outFileCon,append=TRUE)
  cat(sprintf('age_group="%s"\n',results$age_group[i]),file= outFileCon,append=TRUE)
  cat(sprintf('gender="%s"\n',results$gender[i]),file= outFileCon,append=TRUE)
  cat(sprintf('extrovert="%s"\n',results$extrovert[i]),file= outFileCon,append=TRUE)
  cat(sprintf('neurotic="%s"\n',results$neurotic[i]),file= outFileCon,append=TRUE)
  cat(sprintf('agreeable="%s"\n',results$agreeable[i]),file= outFileCon,append=TRUE)
  cat(sprintf('conscientious="%s"\n',results$conscientious[i]),file= outFileCon,append=TRUE)
  cat(sprintf('open="%s"\n',results$open[i]),file= outFileCon,append=TRUE)
  
  cat(sprintf('/>\n'),file= outFileCon,append=TRUE)

  close(outFileCon)

}
##################### END MARKER #################################


#
# Beginning of un-used models for the final predictions
# These were used as part of earlier score cards, but now they are un-used
# as we obtained improved results with above models. 
#
unused_function = function() {
	#
	# Create features using TF-IDF for predicting personality traits
	#
	term_documentTfIdf= DocumentTermMatrix(corpus_clean, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
	SparseRemoved = as.matrix((removeSparseTerms(term_documentTfIdf, sparse = 0.90)))
	dict_train = colnames(SparseRemoved)

	term_documentTfIdf_ptest= DocumentTermMatrix(corpus_clean_ptest, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE), dictionary=dict_train))
	#SparseRemoved_ptest = as.matrix((removeSparseTerms(term_documentTfIdf_ptest, sparse = 0.90)))

	su_train = as.data.frame(SparseRemoved)
	su_test = as.data.frame(as.matrix(term_documentTfIdf_ptest))


	  # The following code predicts age and gender using NB and KNN based on TF-IDF features
	  # The results from these algos is inferior compared to the predictions done using Unigram model
	  # hence these are un-used. 
	  nb_classifier_genderclass = naiveBayes(su_train, profile.raw_train$genderclass)
	  su_test_pred_genderclass = predict(nb_classifier_genderclass, su_test)
	  #confusionMatrix(su_test_pred_genderclass, profile.raw_test$genderclass)
	  
	  nb_classifier_ageclass = naiveBayes(su_train, profile.raw_train$ageclass)
	  su_test_pred_ageclass = predict(nb_classifier_ageclass, su_test)
	  #confusionMatrix(su_test_pred_ageclass, profile.raw_test$ageclass)
	  
	  #knn.genderclass.pred = knn(train=su_train, test=su_test, cl=profile.raw_train$genderclass, k=81)
	  #confusionMatrix(knn.genderclass.pred, profile.raw_test$genderclass)
	  
	  #knn.ageclass.pred = knn(train=su_train, test=su_test, cl=profile.raw_train$ageclass, k=81)
	  #confusionMatrix(knn.ageclass.pred, profile.raw_test$ageclass)

	MAE <- function(predicted, actual) {
	  mean(abs(actual-predicted))
	}

	#Note: The results seem better with linear regression for TF-IDF than Random Forests
	#Note2: Using LIWC the results are better with Random Forests Regression than Linear Regression
	if(useRandomForests == FALSE){
	  #
	  # Openess Model
	  #
	  ope_train_frame = su_train 
	  ope_test_frame = su_test
	  ope_train_frame$ope = profile.raw_train$ope
	  ope_test_frame$ope = profile.raw_ptest$ope
	  
	  lm_classifier_ope = lm(ope ~ ., data = ope_train_frame)
	  lm_pred_ope = predict(lm_classifier_ope, ope_test_frame)
	  #rmse(lm_pred_ope, profile.raw_ptest$ope)
	  #MAE(lm_pred_ope, profile.raw_ptest$ope)
	  
	  #
	  # Conscientiousness Model
	  #
	  con_train_frame = su_train 
	  con_test_frame = su_test
	  con_train_frame$con = profile.raw_train$con
	  con_test_frame$con = profile.raw_ptest$con
	  
	  lm_classifier_con = lm(con ~ ., data = con_train_frame)
	  lm_pred_con = predict(lm_classifier_con, con_test_frame)
	  #rmse(lm_pred_con, profile.raw_ptest$con)
	  #MAE(lm_pred_con, profile.raw_ptest$con)
	  
	  #
	  # Extroversion  Model
	  #
	  ext_train_frame = su_train 
	  ext_test_frame = su_test
	  ext_train_frame$ext = profile.raw_train$ext
	  ext_test_frame$ext = profile.raw_ptest$ext
	  
	  lm_classifier_ext = lm(ext ~ ., data = ext_train_frame)
	  lm_pred_ext = predict(lm_classifier_ext, ext_test_frame)
	  #rmse(lm_pred_ext, profile.raw_ptest$ext)
	  #MAE(lm_pred_ext, profile.raw_ptest$ext)
	  
	  
	  #
	  # Agreeableness Model
	  #
	  agr_train_frame = su_train 
	  agr_test_frame = su_test
	  agr_train_frame$agr = profile.raw_train$agr
	  agr_test_frame$agr = profile.raw_ptest$agr
	  
	  lm_classifier_agr = lm(agr ~ ., data = agr_train_frame)
	  lm_pred_agr = predict(lm_classifier_agr, agr_test_frame)
	  #rmse(lm_pred_agr, profile.raw_ptest$agr)
	  #MAE(lm_pred_agr, profile.raw_ptest$agr)
	  
	  
	  #
	  # Emotional Stability  Model
	  #
	  neu_train_frame = su_train 
	  neu_test_frame = su_test
	  neu_train_frame$neu = profile.raw_train$neu
	  neu_test_frame$neu = profile.raw_ptest$neu
	  
	  lm_classifier_neu = lm(neu ~ ., data = neu_train_frame)
	  lm_pred_neu = predict(lm_classifier_neu, neu_test_frame)
	  #rmse(lm_pred_neu, profile.raw_ptest$neu)
	  #MAE(lm_pred_neu, profile.raw_ptest$neu)
	  
	  
	  results = data.frame(userId = profile.raw_ptest$userid, age_group = as.character(su_test_pred_ageclass_laplace), 
						   gender = as.character(su_test_pred_genderclass_laplace), 
						   extrovert = lm_pred_ext,
						   neurotic = lm_pred_neu, 
						   agreeable = lm_pred_agr,
						   conscientious = lm_pred_con, 
						   open = lm_pred_ope,  stringsAsFactors = FALSE)
	  
	} else {
	  
	  # Prepare Data for using with randomForest Model
	  library(randomForest)
	  colnames(su_train)<-paste(rep("col",ncol(su_train)),c(1:ncol(su_train)),sep="")
	  colnames(su_test)<-paste(rep("col",ncol(su_test)),c(1:ncol(su_test)),sep="")
	  rownames(su_train) = profile.raw_train$userid
	  rownames(su_test) =  profile.raw_ptest$userid
	  
	  ope_train_frame = su_train
	  ope_test_frame = su_test
	  ope_train_frame$ope = profile.raw_train$ope
	  ope_test_frame$ope = profile.raw_ptest$ope
	  
	  rf.ope = randomForest(ope ~ ., data = ope_train_frame, mtry=9, importance=TRUE, ntree=100)
	  rf.ope.pred = predict(rf.ope, ope_test_frame)
	  #rmse(rf.ope.pred, profile.raw_ptest$ope)
	  #MAE(rf.ope.pred, profile.raw_ptest$ope)
	  
	  con_train_frame = su_train
	  con_test_frame = su_test
	  con_train_frame$con = profile.raw_train$con
	  con_test_frame$con = profile.raw_ptest$con
	  
	  rf.con = randomForest(con ~ ., data = con_train_frame, mtry=9, importance=TRUE, ntree=100)
	  rf.con.pred = predict(rf.con, con_test_frame)
	  #rmse(rf.con.pred, profile.raw_ptest$con)
	  #MAE(rf.con.pred, profile.raw_ptest$con)
	  
	  ext_train_frame = su_train
	  ext_test_frame = su_test
	  ext_train_frame$ext = profile.raw_train$ext
	  ext_test_frame$ext = profile.raw_ptest$ext
	  
	  rf.ext = randomForest(ext ~ ., data = ext_train_frame, mtry=9, importance=TRUE, ntree=100)
	  rf.ext.pred = predict(rf.ext, ext_test_frame)
	  #rmse(rf.ext.pred, profile.raw_ptest$ext)
	  #MAE(rf.ext.pred, profile.raw_ptest$ext)
	  
	  agr_train_frame = su_train
	  agr_test_frame = su_test
	  agr_train_frame$agr = profile.raw_train$agr
	  agr_test_frame$agr = profile.raw_ptest$agr
	  
	  rf.agr = randomForest(agr ~ ., data = agr_train_frame, mtry=9, importance=TRUE, ntree=100)
	  rf.agr.pred = predict(rf.agr, agr_test_frame)
	  #rmse(rf.agr.pred, profile.raw_ptest$agr)
	  #MAE(rf.agr.pred, profile.raw_ptest$agr)
	  
	  neu_train_frame = su_train
	  neu_test_frame = su_test
	  neu_train_frame$neu = profile.raw_train$neu
	  neu_test_frame$neu = profile.raw_ptest$neu
	  
	  rf.neu = randomForest(neu ~ ., data = neu_train_frame, mtry=9, importance=TRUE, ntree=100)
	  rf.neu.pred = predict(rf.neu, neu_test_frame)
	  #rmse(rf.neu.pred, profile.raw_ptest$neu)
	  #MAE(rf.neu.pred, profile.raw_ptest$neu)  
	  
	  results = data.frame(userId = profile.raw_ptest$userid, age_group = as.character(su_test_pred_ageclass_laplace), 
						   gender = as.character(su_test_pred_genderclass_laplace), 
						   extrovert = rf.ext.pred,
						   neurotic = rf.neu.pred, 
						   agreeable = rf.agr.pred,
						   conscientious = rf.con.pred, 
						   open = rf.con.ope,  stringsAsFactors = FALSE)
	}

} # End of un-used function


#
# Beginning of experimental models that were done as part of the project. 
# They were never used in the scorecard generation. But we have used them
# to collect the results using Stratified Random Sampling and K-fold Cross Validation
# The models with the better results were used in the final score card generation
#
unused_function2 = function() {

	TrainingDir = "/home/radhi/assignments/mlproject/TCSS555/Train"
	
	#
	# Code for exploring data properties
	#

	# Check the data spread and centrality of the training data using boxplots and histograms
	#hist(profile.raw_test$ope)
	#hist(profile.raw_test$con)
	#hist(profile.raw_test$agr)
	#hist(profile.raw_test$ext)
	#hist(profile.raw_test$neu)
	#hist(profile.raw_test$age)
	boxplot(profile.raw_test$ope, main="Boxplot of Openness", ylab = "Openeess Score between [1; 5]", varwidth = TRUE)
	boxplot(profile.raw_test$con, main="Boxplot of Conscientiousness", ylab = "Conscientiousness Score between [1; 5]")
	boxplot(profile.raw_test$agr, main="Boxplot of Agreeableness", ylab = "Agreeableness Score between [1; 5]")
	boxplot(profile.raw_test$ext, main="Boxplot of Extroversion", ylab = "Extroversion Score between [1; 5]")
	boxplot(profile.raw_test$neu, main="Boxplot of Emotional Stability", ylab = "Emotional Stability Score between [1; 5]")

	# check the class proportions for Age and Gender Classes
	age_table = table(profile.raw_test$ageclass)
	gender_table = table(profile.raw_test$genderclass)
	prop.table(age_table)
	prop.table(gender_table)

	# Explore the relationship using correlation matrix
	#cor(profile.raw_test[c("age","gender","ope","con","ext","agr","neu" )])

	# we can also use scatterplot to visualize the relationship between two variables
	#plot(profile.raw_test$age, profile.raw_test$ope)
	#plot(profile.raw_test$age, profile.raw_test$con)

	# Visualize the relationships using the scatterplot matrix
	#install.packages("psych")
	#library(psych)
	#pairs.panels(profile.raw_test[c("age","gender","ope","con","ext","agr","neu" )])
	
	#
	# Code for generating Word Cloud of Status Updates in Training Corpus
	#
	install.packages("wordcloud")
	library(wordcloud)
	wordcloud(corpus_clean, min.freq=100, random.order=FALSE)
	
	#
	# Code for Generating Plots for Top10 Important LIWC features for each trait. 
	#
	
	#x=varImp(rf.liwc.ope)
	#ope.imp = data.frame(Features=rownames(x), Overall = x,  stringsAsFactors = FALSE)
	#head(arrange(ope.imp, Overall, decreasing = TRUE), n=10L)
	varImpPlot(rf.liwc.ope, type=1, n.var=10,main="Top 10 Important LIWC Features for Predicting Openness")
	varImpPlot(rf.liwc.con, type=1, n.var=10,main="Top 10 Important LIWC Features for Predicting Conscientiousness")
	varImpPlot(rf.liwc.agr, type=1, n.var=10,main="Top 10 Important LIWC Features for Predicting Agreeableness")
	varImpPlot(rf.liwc.ext, type=1, n.var=10,main="Top 10 Important LIWC Features for Predicting Extroversion")
	varImpPlot(rf.liwc.neu, type=1, n.var=10,main="Top 10 Important LIWC Features for Predicting Emotional Stability")

	#
	# Code for generating AUC and ROC Curves For Gender model
	#
	install.packages("ROCR")
	library(ROCR)
	gender_prob = predict(nb_classifier_genderclass_laplace, su_test, type="raw")
	gender_pred = prediction(predictions=gender_prob[,2], labels = profile.raw_test$genderclass,  label.ordering = NULL)
	gender_perf = performance(gender_pred, measure="tpr", x.measure="fpr")
	plot(gender_perf, main="ROC Curve for Gender Model", col="blue", lwd=3)
	abline(a=0,b=1,lwd=2,lty=2)
	gender_auc = performance(gender_pred, measure="auc")
	unlist(gender_auc@y.values)
	
	# Note: ROCR currently does not support plots for multi-class variables => This cannot be done for Age model

	# read LIWC.csv
	liwcfeatures = read.csv(paste(TrainingDir, "LIWC.csv", sep="/"), stringsAsFactors=FALSE)
	liwcfeatures$userid = liwcfeatures$userId
	liwcfeatures = liwcfeatures[,-1]  # remove userId column and retain "userid" column
	profile.raw_train = merge(liwcfeatures, profile.raw_train, by="userid")

	#Stratified Random Sampling
	set.seed(123)  # Fixing the seed so that we get consistent results at different times
	in_train = createDataPartition(profile.raw_train$genderclass, p=0.80, list=FALSE)

	# Partition into Training and Hold-out sets based on Stratified Random Sampling
	liwc.train <- profile.raw_train[in_train,]
	liwc.vs <- profile.raw_train[-in_train,] 

	profile.raw_train1 <- profile.raw_train[in_train,]
	profile.raw_test <- profile.raw_train[-in_train,]
	
	su_corpus_train = corpus_clean[in_train]
	su_corpus_test = corpus_clean[-in_train]
	
	#
	# The following code predicts age and gender using TF.NB i.e. Term Frequency with Naive Bayes
	# Note: The results from these this are inferior compared to the predictions done using NB.Bin (Unigram model)
	# and hence this is un-used in the final scorecard generation 
	#
	{
	  su_train1 = DocumentTermMatrix(su_corpus_train, list(dictionary=su_dict_train))
	  su_test1 = DocumentTermMatrix(su_corpus_test, list(dictionary=su_dict_train))
	  su_train1 = as.data.frame(as.matrix(su_train1))
	  su_test1 = as.data.frame(as.matrix(su_test1))
	  
	  nb_classifier_genderclass_laplace1 = naiveBayes(su_train1, profile.raw_train1$genderclass, laplace = 1)
	  su_test_pred_genderclass_laplace1 = predict(nb_classifier_genderclass_laplace1, su_test1)
	  #CrossTable(su_test_pred_genderclass_laplace, profile.raw_test$genderclass, prop.chisq = FALSE, prop.t=FALSE, dnn = c('predicted', 'actual'))
	  confusionMatrix(su_test_pred_genderclass_laplace1, profile.raw_test$genderclass)  
	  
	  nb_classifier_ageclass_laplace1 = naiveBayes(su_train1, profile.raw_train1$ageclass)
	  su_test_pred_ageclass_laplace1 = predict(nb_classifier_ageclass_laplace1, su_test1)
	  #CrossTable(su_test_pred_ageclass_laplace, profile.raw_test$ageclass, prop.chisq = FALSE, prop.t=FALSE, dnn = c('predicted', 'actual'))
	  confusionMatrix(su_test_pred_ageclass_laplace1, profile.raw_test$ageclass)
	  
	}

	#
	# The following code predicts age and gender using TFIDF.NB
	# Note: The results from these this are inferior compared to the predictions done using NB.Bin (Unigram model)
	# and hence this is un-used in the final scorecard generation 
	#
	{
	  
	  term_documentTfIdf= DocumentTermMatrix(corpus_clean, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
	  SparseRemoved = as.matrix((removeSparseTerms(term_documentTfIdf, sparse = 0.90)))
	  #sum(rowSums(as.matrix(SparseRemoved)) == 0)
	  #ncol(SparseRemoved)
	  #colnames(SparseRemoved)
	  
	  su_dtm_train_tfidf = SparseRemoved[in_train,]
	  su_dtm_test_tfidf = SparseRemoved[-in_train,]
	  
	  su_train_tfidf = as.data.frame(su_dtm_train_tfidf)
	  su_test_tfidf = as.data.frame(su_dtm_test_tfidf)
	  
	  nb_classifier_genderclass_tfidf = naiveBayes(su_train_tfidf, profile.raw_train1$genderclass)
	  su_test_pred_genderclas_tfidfs = predict(nb_classifier_genderclass_tfidf, su_test_tfidf)
	  confusionMatrix(su_test_pred_genderclas_tfidfs, profile.raw_test$genderclass)
	  
	  nb_classifier_ageclass_tfidf = naiveBayes(su_train_tfidf, profile.raw_train1$ageclass)
	  su_test_pred_ageclass_tfidf = predict(nb_classifier_ageclass_tfidf, su_test_tfidf)
	  confusionMatrix(su_test_pred_ageclass_tfidf, profile.raw_test$ageclass)  
	  
	}

	##
	# Age and Gender Prediction using LIWC Features and Naive Bayes (LIWC.NB)
	##
	nb.genderclass = naiveBayes(liwc.train[,3:83], liwc.train$genderclass)
	nb.genderclass.pred = predict(nb.genderclass, liwc.vs[,3:83])
	confusionMatrix(nb.genderclass.pred, liwc.vs$genderclass)

	nb.ageclass = naiveBayes(liwc.train[,3:83], liwc.train$ageclass)
	nb.ageclass.pred = predict(nb.ageclass, liwc.vs[,3:83])
	confusionMatrix(nb.ageclass.pred, liwc.vs$ageclass)

	##
	# Age and Gender Prediction using LIWC Features and randomForest Classifier (LIWC.RF)
	##
	liwc.genderclass.train = liwc.train[,3:83]
	liwc.genderclass.train$genderclass = liwc.train$genderclass
	liwc.genderclass.vs = liwc.vs[,3:83]
	liwc.genderclass.vs$genderclass = liwc.vs$genderclass

	rf.liwc.genderclass = randomForest(genderclass ~ ., data = liwc.genderclass.train, mtry=81, importance=TRUE , ntree=100)
	rf.liwc.genderclass.pred = predict(rf.liwc.genderclass, liwc.genderclass.vs)
	confusionMatrix(rf.liwc.genderclass.pred, liwc.genderclass.vs$genderclass)

	liwc.ageclass.train = liwc.train[,3:83]
	liwc.ageclass.train$ageclass = liwc.train$ageclass
	liwc.ageclass.vs = liwc.vs[,3:83]
	liwc.ageclass.vs$ageclass = liwc.vs$ageclass

	rf.liwc.ageclass = randomForest(ageclass ~ ., data = liwc.ageclass.train, mtry=81, importance=TRUE , ntree=100)
	rf.liwc.ageclass.pred = predict(rf.liwc.ageclass, liwc.ageclass.vs)
	confusionMatrix(rf.liwc.ageclass.pred, liwc.ageclass.vs$ageclass)

	##
	# Age and Gender Prediction using LIWC Features and SVM Radial Kernel (LIWC.SVM)
	##
	#gender prediction using SVM radial classifier
	svm.liwc.genderclass = svm(genderclass ~ ., data = liwc.genderclass.train, kernel = "radial", cost=1, gamma=0.01, scale=TRUE)
	svm.liwc.genderclass.pred = predict(svm.liwc.genderclass, liwc.genderclass.vs)
	confusionMatrix(svm.liwc.genderclass.pred, liwc.genderclass.vs$genderclass)

	# age prediction using SVM radial classifier
	svm.liwc.ageclass = svm(ageclass ~ ., data = liwc.ageclass.train, kernel = "radial", cost=1, gamma=0.01, scale=TRUE)
	svm.liwc.ageclass.pred = predict(svm.liwc.ageclass, liwc.ageclass.vs)
	confusionMatrix(svm.liwc.ageclass.pred, liwc.ageclass.vs$ageclass)

	##
	# Age and Gender Prediction using Combination of LIWC Features/TFIDF and Naive Bayes
	##
	{
	  
	  su_train_tfidf = as.data.frame(su_dtm_train_tfidf)
	  su_train_tfidf$userid = profile.raw_train1$userid
	  su_test_tfidf = as.data.frame(su_dtm_test_tfidf)
	  su_test_tfidf$userid = profile.raw_test$userid
	  
	  su_train_tfidf1 = merge(su_train_tfidf, liwc.train, by="userid")
	  su_test_tfidf1 = merge(su_test_tfidf, liwc.vs, by="userid")
	  
	  feature_col = c(2:96, 98:178)  #2:96 indicate the tfidf, 98:178 indicate the LIWC features after merge
	  

	  nb_classifier_genderclass_tfidf1 = naiveBayes(su_train_tfidf1[,feature_col], profile.raw_train1$genderclass)
	  su_test_pred_genderclas_tfidfs1 = predict(nb_classifier_genderclass_tfidf1, su_test_tfidf1[,feature_col])
	  confusionMatrix(su_test_pred_genderclas_tfidfs1, profile.raw_test$genderclass)
	  
	  nb_classifier_ageclass_tfidf1 = naiveBayes(su_train_tfidf1[,feature_col], profile.raw_train1$ageclass)
	  su_test_pred_ageclas_tfidfs1 = predict(nb_classifier_ageclass_tfidf1, su_test_tfidf1[,feature_col])
	  confusionMatrix(su_test_pred_ageclas_tfidfs1, profile.raw_test$ageclass)
	}	
	
	##
	# Personality Traits using LIWC features
	##

	#
	# Openess Model
	#
	liwc.ope.train = liwc.train[,3:83]
	liwc.ope.train$ope = liwc.train$ope
	liwc.ope.vs = liwc.vs[,3:83]
	liwc.ope.vs$ope = liwc.vs$ope

	rf.liwc.ope = randomForest(ope ~ ., data = liwc.ope.train, mtry=9, importance=TRUE , ntree=100)
	rf.liwc.ope.pred = predict(rf.liwc.ope, liwc.ope.vs)
	rmse(rf.liwc.ope.pred, liwc.ope.vs$ope)

	#
	# Conscientiousness Model
	#
	liwc.con.train = liwc.train[,3:83]
	liwc.con.train$con = liwc.train$con
	liwc.con.vs = liwc.vs[,3:83]
	liwc.con.vs$con = liwc.vs$con

	rf.liwc.con = randomForest(con ~ ., data = liwc.con.train, mtry=9, importance=TRUE , ntree=100)
	rf.liwc.con.pred = predict(rf.liwc.con, liwc.con.vs)
	rmse(rf.liwc.con.pred, liwc.con.vs$con)

	#
	# Extroversion  Model
	#
	liwc.ext.train = liwc.train[,3:83]
	liwc.ext.train$ext = liwc.train$ext
	liwc.ext.vs = liwc.vs[,3:83]
	liwc.ext.vs$ext = liwc.vs$ext

	rf.liwc.ext = randomForest(ext ~ ., data = liwc.ext.train, mtry=9, importance=TRUE , ntree=100)
	rf.liwc.ext.pred = predict(rf.liwc.ext, liwc.ext.vs)
	rmse(rf.liwc.ext.pred, liwc.ext.vs$ext)

	#
	# Agreeableness Model
	#
	liwc.agr.train = liwc.train[,3:83]
	liwc.agr.train$agr = liwc.train$agr
	liwc.agr.vs = liwc.vs[,3:83]
	liwc.agr.vs$agr = liwc.vs$agr

	rf.liwc.agr = randomForest(agr ~ ., data = liwc.agr.train, mtry=9, importance=TRUE , ntree=100)
	rf.liwc.agr.pred = predict(rf.liwc.agr, liwc.agr.vs)
	rmse(rf.liwc.agr.pred, liwc.agr.vs$agr)

	#
	# Emotional Stability  Model
	#
	liwc.neu.train = liwc.train[,3:83]
	liwc.neu.train$neu = liwc.train$neu
	liwc.neu.vs = liwc.vs[,3:83]
	liwc.neu.vs$neu = liwc.vs$neu

	rf.liwc.neu = randomForest(neu ~ ., data = liwc.neu.train, mtry=9, importance=TRUE , ntree=100)
	rf.liwc.neu.pred = predict(rf.liwc.neu, liwc.neu.vs)
	rmse(rf.liwc.neu.pred, liwc.neu.vs$neu)
	
	##############################################################
	### Code for K-Fold Cross Validation
	##############################################################
	su_dict_train_kfold = findFreqTerms(su_dtm_train,90) 
	su_dtm_kfold = DocumentTermMatrix(corpus_clean, list(dictionary=su_dict_train_kfold))
	train_kfold = as.data.frame(as.matrix(su_dtm_kfold))
	train_kfold = apply(train_kfold, MARGIN = 2, convert_counts)	
	
	# set up the control parameter to train with the 10-fold cross validation
	control = trainControl(method="cv", number=10)
	#setup control parameter to train with repeated 10-fold cross validation for 3 times. 
	#control1 = trainControl(method="repeatedcv", number=10, repeats=3)
	
	# 10-fold CV for NB.Bin For Gender
	nb_gender_kfold_fit = train(train_kfold, profile.raw$genderclass, method = "nb",trControl=control)
	kfold_pred_gender = predict(nb_gender_kfold_fit, train_kfold)
	confusionMatrix(kfold_pred_gender, profile.raw$genderclass)

	# 10-fold CV for NB.Bin For Age
	nb_age_kfold_fit = train(train_kfold, profile.raw$ageclass, method = "nb",trControl=control)
	kfold_pred_age = predict(nb_age_kfold_fit, train_kfold)
	confusionMatrix(kfold_pred_age, profile.raw$ageclass)
	
	# 10-fold CV based on TFIDF for different ML Algos
	train_kfold_tfidf = as.data.frame(SparseRemoved)

	lm_ope_kfold_fit = train(train_kfold_tfidf, profile.raw$ope, method = "lm",trControl=control)
	lm_kfold_pred_ope = predict(lm_ope_kfold_fit$finalModel, train_kfold_tfidf)
	rmse(lm_kfold_pred_ope, profile.raw$ope)

	rf_ope_kfold_fit = train(train_kfold_tfidf, profile.raw$ope, method = "rf", tuneGrid=data.frame(mtry=9),ntree=100, importance=TRUE, trControl=trainControl(method="cv", number=5))
	rf_kfold_pred_ope = predict(rf_ope_kfold_fit$finalModel, newdata = train_kfold_tfidf)
	rmse(rf_kfold_pred_ope, profile.raw$ope)

	lm_con_kfold_fit = train(train_kfold_tfidf, profile.raw$con, method = "lm",trControl=control)
	lm_kfold_pred_con = predict(lm_con_kfold_fit$finalModel, train_kfold_tfidf)
	rmse(kfold_pred_con, profile.raw$con)

	rf_con_kfold_fit = train(train_kfold_tfidf, profile.raw$con, method = "rf", tuneGrid=data.frame(mtry=9),ntree=100, importance=TRUE, trControl=trainControl(method="cv", number=5))
	rf_kfold_pred_con = predict(rf_con_kfold_fit$finalModel, train_kfold_tfidf)
	rmse(rf_kfold_pred_con, profile.raw$con)

	lm_ext_kfold_fit = train(train_kfold_tfidf, profile.raw$ext, method = "lm",trControl=control)
	lm_kfold_pred_ext = predict(lm_ext_kfold_fit$finalModel, train_kfold_tfidf)
	rmse(lm_kfold_pred_ext, profile.raw$ext)

	rf_ext_kfold_fit = train(train_kfold_tfidf, profile.raw$ext, method = "rf", tuneGrid=data.frame(mtry=9),ntree=100, importance=TRUE, trControl=trainControl(method="cv", number=5))
	rf_kfold_pred_ext = predict(rf_ext_kfold_fit$finalModel, train_kfold_tfidf)
	rmse(rf_kfold_pred_ext, profile.raw$ext)

	lm_agr_kfold_fit = train(train_kfold_tfidf, profile.raw$agr, method = "lm",trControl=control)
	lm_kfold_pred_agr = predict(lm_agr_kfold_fit$finalModel, train_kfold_tfidf)
	rmse(lm_kfold_pred_agr, profile.raw$agr)

	rf_agr_kfold_fit = train(train_kfold_tfidf, profile.raw$agr, method = "rf", tuneGrid=data.frame(mtry=9),ntree=100, importance=TRUE, trControl=trainControl(method="cv", number=5))
	rf_kfold_pred_agr = predict(rf_agr_kfold_fit$finalModel, train_kfold_tfidf)
	rmse(rf_kfold_pred_agr, profile.raw$agr)

	lm_neu_kfold_fit = train(train_kfold_tfidf, profile.raw$neu, method = "lm",trControl=control)
	lm_kfold_pred_neu = predict(lm_neu_kfold_fit$finalModel, train_kfold_tfidf)
	rmse(lm_kfold_pred_neu, profile.raw$neu)

	rf_neu_kfold_fit = train(train_kfold_tfidf, profile.raw$neu, method = "rf", tuneGrid=data.frame(mtry=9),ntree=100, importance=TRUE, trControl=trainControl(method="cv", number=5))
	rf_kfold_pred_neu = predict(rf_neu_kfold_fit$finalModel, train_kfold_tfidf)
	rmse(rf_kfold_pred_neu, profile.raw$neu)
	#plot(rf_neu_kfold_fit$finalModel)


	nb_genderclass_kfold_fit = train(train_kfold_tfidf, profile.raw$genderclass, method = "nb",trControl=control)
	#print(nb_genderclass_kfold_fit)
	nb_kfold_pred_genderclass = predict(nb_genderclass_kfold_fit$finalModel, train_kfold_tfidf)
	confusionMatrix(nb_kfold_pred_genderclass$class, profile.raw$genderclass)
	#plot the features with Naive Bayes
	#plot(nb_genderclass_kfold_fit$finalModel)

	nb_ageclass_kfold_fit = train(train_kfold_tfidf, profile.raw$ageclass, method = "nb",trControl=control) 
	nb_kfold_pred_ageclass = predict(nb_ageclass_kfold_fit$finalModel, train_kfold_tfidf)
	confusionMatrix(nb_kfold_pred_ageclass$class, profile.raw$ageclass)
	#plot(nb_ageclass_kfold_fit$finalModel)

	rf_genderclass_kfold_fit = train(train_kfold_tfidf, profile.raw$genderclass, method = "rf", tuneGrid=data.frame(mtry=9),ntree=100, importance=TRUE, trControl=trainControl(method="cv", number=10))
	rf_kfold_pred_genderclass = predict(rf_genderclass_kfold_fit$finalModel, train_kfold_tfidf)
	confusionMatrix(rf_kfold_pred_genderclass, profile.raw$genderclass)
	#plot(rf_genderclass_kfold_fit$finalModel)

	rf_ageclass_kfold_fit = train(train_kfold_tfidf, profile.raw$ageclass, method = "rf", tuneGrid=data.frame(mtry=9),ntree=100, importance=TRUE, trControl=trainControl(method="cv", number=10))
	rf_kfold_pred_ageclass = predict(rf_ageclass_kfold_fit$finalModel, train_kfold_tfidf)
	confusionMatrix(rf_kfold_pred_ageclass, profile.raw$ageclass)
	

}# End of un-used function2

