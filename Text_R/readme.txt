This part of the project is designed and implemented by Radha M. 
I chose to implement the project using R and Status Updates. 

Summary of the files
===============================================================================
- readme.txt: this file. 
- tcss555: This is a shell script that invokes combined.R using Rscript. 
- combined.R: This is the main script file that extracts the features
from Text and Trains different models to predict age, gender and personality.
Script has hard-coded training path and it is assumed to be in directory 
"/data/training"  with the approriate layout 
ex: i.e. /data/training/profile/profile.csv etc.

Running the programs:
===============================================================================
- Predictions using the above program can be obtained as follows:
  ex1: ./tcss555 -i /data/public-test-data -o /home/itadmin/mockresults/week6
  ex2: ./tcss555 -i /home/radhi/assignments/mlproject/TCSS555/Public_Test 
	-o results/sample
	
- Test data path is specified with -i argument
- Results path is specified with -o path

Note1: The path and file names in Linux are case-sensitive. Test data path is 
expected to have same layout and same case-sensitiveness as was in Shared VM. 

Note2: combined.R script assumes the required packages are already installed 
on the target machine. We can use the following R Studio/Shell based commands
to install the required packages if they are not already installed: 

	install.packages("tm", dependencies = TRUE)
	install.packages("SnowballC", dependencies = TRUE)
	install.packages("readr", dependencies = TRUE)
	install.packages("e1071", dependencies = TRUE)
	install.packages("gmodels", dependencies = TRUE)
	install.packages("randomForest", dependencies = TRUE)
	install.packages("caret", dependencies = TRUE)
	install.packages("class", dependencies = TRUE)
	install.packages("hydroGOF", dependencies = TRUE)
	install.packages("wordcloud", dependencies = TRUE)
	install.packages("ROCR", dependencies = TRUE)

Description
===============================================================================
- Please look into combined.R file to understand the flow as the code is 
written with in-line documentation. 

Misc
===============================================================================
Note: We can use the following commands to find out whether output files 
are correctly written or not and if all the predictions are there or not. 

grep male * | wc
grep female * | wc

grep xx-24 * | wc
grep 25-34 * | wc
grep 35-49 * | wc
grep 50-xx * | wc

grep extrovert * | wc
grep neurotic * | wc
grep agreeable * | wc
grep conscientious * | wc
grep open * | wc
