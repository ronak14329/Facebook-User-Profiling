###########################################################################
# Author: Radha Madhavi Malireddy
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

TrainingDir = "/data/training"

#TestDir = "/home/radhi/assignments/mlproject/TCSS555/Public_Test"
#ResultsDir = "/home/radhi/assignments/mlproject/results/sample"
#TrainingDir = "/home/radhi/assignments/mlproject/TCSS555/Train"

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

# 1. Read the Profile data
profile.raw_train = read.csv(paste(TrainingDir, "profile/profile.csv", sep="/"), stringsAsFactors=FALSE)
profile.raw_ptest = read.csv(paste(TestDir, "profile/profile.csv", sep="/"), stringsAsFactors=FALSE)

#2. Prepare the data

# Based on the structure information profile.raw$gender is read as numeric variable. We need to format
# it as a factor for fitting the training data. 
profile.raw_train$genderclass = as.factor(ifelse(profile.raw_train$gender == 1, "female", "male"))

# Based on the structure information profile.raw$age is read as numeric variable. We need to format
# as a multi-level factor variable for fitting the training data. 
# note: We can also use "findInterval" for bucketizing the age. 
profile.raw_train$ageclass = cut(profile.raw_train$age, 
    breaks = c(-Inf, 25, 35, 50, Inf), 
    labels = c("xx-24", "25-34", "35-49", "50-xx"), 
    right = FALSE)


# Status updates are strings of text composed of words, spaces, numbers, and punctuation.
# We need to remove numbers, punctuation, handle uninterresting words such as and, but and or. 
# all of the text mining processing can be done using "tm" package in R. 

#Create a new column vector profile.raw$statusupdates
for(i in 1:length(profile.raw_train$userid))
{
  profile.raw_train$statusupdates[i] = read_file(paste(paste(TrainingDir, "text", sep="/"), paste(profile.raw_train$userid[i], "txt", sep = "."), sep="/"))
}

for(i in 1:length(profile.raw_ptest$userid))
{
  profile.raw_ptest$statusupdates[i] = read_file(paste(paste(TestDir, "text", sep="/"), paste(profile.raw_ptest$userid[i], "txt", sep = "."), sep="/"))
}

# clean text
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
su_dict_train = findFreqTerms(su_dtm_train, 90) 
su_train = DocumentTermMatrix(corpus_clean, list(dictionary=su_dict_train))
su_test = DocumentTermMatrix(corpus_clean_ptest, list(dictionary=su_dict_train))

# Create indicator features
convert_counts = function (x) {
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return (x)
}

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
library(randomForest)

# read LIWC.csv for Training and Test Data
# Merge into the Profile.csv data
liwc.train = read.csv(paste(TrainingDir, "LIWC.csv", sep="/"), stringsAsFactors=FALSE)
liwc.train$userid = liwc.train$userId
liwc.train = liwc.train[,-1]  # remove userId column and retain "userid" column
liwc.train = merge(liwc.train, profile.raw_train, by="userid")
liwc.vs = read.csv(paste(TestDir, "LIWC.csv", sep="/"), stringsAsFactors=FALSE)
liwc.vs$userid = liwc.vs$userId
liwc.vs = liwc.vs[,-1]  # remove userId column and retain "userid" column
liwc.vs = merge(liwc.vs, profile.raw_ptest, by="userid")


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

rf.liwc.combined = data.frame(userId = liwc.vs$userid,
                              extrovert = rf.liwc.ext.pred,
                              neurotic = rf.liwc.neu.pred, 
                              agreeable = rf.liwc.agr.pred,
                              conscientious = rf.liwc.con.pred, 
                              open = rf.liwc.ope.pred,  stringsAsFactors = FALSE)

nb.agegender.combined = data.frame(userId = profile.raw_ptest$userid, 
                                   age_group = as.character(su_test_pred_ageclass_laplace),
                                   gender = as.character(su_test_pred_genderclass_laplace),
                                   stringsAsFactors = FALSE)
#merge the results
results = merge(rf.liwc.combined, nb.agegender.combined , by="userId")


#
# Beginning of Un-used features or models for the final predictions
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

for (i in 1:length(results$userId)) {
  outFileName = paste(ResultsDir, paste(results$userId[i], "xml", sep = "."), sep="/")

  #cat('<?xml version="1.0" encoding="UTF-8"?>\n', file= outFileName)
    
  cat(sprintf('<user \n'),file= outFileName)    
  cat(sprintf('userId="%s"\n',results$userId[i]),file= outFileName,append=TRUE)
  cat(sprintf('age_group="%s"\n',results$age_group[i]),file= outFileName,append=TRUE)
  cat(sprintf('gender="%s"\n',results$gender[i]),file= outFileName,append=TRUE)
  cat(sprintf('extrovert="%s"\n',results$extrovert[i]),file= outFileName,append=TRUE)
  cat(sprintf('neurotic="%s"\n',results$neurotic[i]),file= outFileName,append=TRUE)
  cat(sprintf('agreeable="%s"\n',results$agreeable[i]),file= outFileName,append=TRUE)
  cat(sprintf('conscientious="%s"\n',results$conscientious[i]),file= outFileName,append=TRUE)
  cat(sprintf('open="%s"\n',results$open[i]),file= outFileName,append=TRUE)
  
  cat(sprintf('/>\n'),file= outFileName,append=TRUE)
}
