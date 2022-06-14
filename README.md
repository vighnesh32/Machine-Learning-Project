# Machine-Learning-Project

Steps for running the code scripts.

       SOFTWARE VERSIONS USED:
MATLAB
Version: R2021a Update 5 (9.10.0.1739362)
	 64-bit (Win64)
	 August 9, 2021


 File Name	  	       Description

1.diabetes.csv            Original data set

2.diabholdout.m           MATLAB script of descriptive statistics,EDA,partition of original data 
		          set into train and test sets (holdout method), Decision Tree model training
                          and testing, Naive Bayes model training and testing.

2.diabkfold.m             MATLAB script of descriptive statistics,EDA,partition of original data 
		          set into train and test sets (kfold method), Decision Tree model training
                          and testing, Naive Bayes model training and testing.   

3.hyperparameter.m        MATLAB script of optimization of the models.

4.testingdataDT.csv       Excel file of test data used for Decision tree model.

5.testingdataNB.csv       Excel file of test data used foe Naive Bayes model.


          Sequence of running the codes:

1. Please start by opening and running each section individually of the file 'diabholdout.m' then run 'hyperparameter.m'
   section by section for optimization.

2. After this for kfold method please run file "kfold.m" then run 'hyperparameter.m' for optimization.

