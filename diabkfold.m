% read data
clc
clear
close all
warning off
%% 
% dataset link - https://www.kaggle.com/uciml/pima-indians-diabetes-database
% reference - https://uk.mathworks.com/help/matlab/ref/readtable.html
df_dia = readtable('diabetes.csv'); 
%% 
% displaying the portion(head) of dataset
% reference - https://uk.mathworks.com/help/matlab/ref/table.head.html
head(df_dia)
%% 
% checking for missing values in dataset
% reference - https://uk.mathworks.com/help/matlab/matlab_prog/test-for-empty-strings-and-missing-values.html#TestForEmptyStringsAndMissingValuesExample-3
sum(ismissing(df_dia))
%% 
% normalizing certain columns of the dataset
% reference - https://uk.mathworks.com/help/matlab/ref/double.normalize.html#:~:text=Description,-example&text=N%20%3D%20normalize(%20A%20)%20returns,each%20column%20of%20data%20separately.
%Normalizing Pregnancies column
df_dia.Pregnancies = df_dia.Pregnancies - min(df_dia.Pregnancies(:));
df_dia.Pregnancies = df_dia.Pregnancies ./ max(df_dia.Pregnancies(:));

% normalizing Glucose column
df_dia.Glucose = df_dia.Glucose - min(df_dia.Glucose(:));
df_dia.Glucose = df_dia.Glucose ./ max(df_dia.Glucose(:));

% normalizing BloodPressure column
df_dia.BloodPressure = df_dia.BloodPressure - min(df_dia.BloodPressure(:));
df_dia.BloodPressure = df_dia.BloodPressure ./ max(df_dia.BloodPressure(:));

% normalizing SkinThickness column
df_dia.SkinThickness = df_dia.SkinThickness - min(df_dia.SkinThickness(:));
df_dia.SkinThickness = df_dia.SkinThickness ./ max(df_dia.SkinThickness(:));

% normalizing Insulin column
df_dia.Insulin = df_dia.Insulin - min(df_dia.Insulin(:));
df_dia.Insulin = df_dia.Insulin ./ max(df_dia.Insulin(:));

% normalizing BMI column
df_dia.BMI = df_dia.BMI - min(df_dia.BMI(:));
df_dia.BMI = df_dia.BMI ./ max(df_dia.BMI(:));

% normalizing DiabetesPedigreeFunction column
df_dia.DiabetesPedigreeFunction = df_dia.DiabetesPedigreeFunction - min(df_dia.DiabetesPedigreeFunction(:));
df_dia.DiabetesPedigreeFunction = df_dia.DiabetesPedigreeFunction ./ max(df_dia.DiabetesPedigreeFunction(:));

% normalizing Age column
df_dia.Age = df_dia.Age - min(df_dia.Age(:));
df_dia.Age = df_dia.Age ./ max(df_dia.Age(:));

% printing dataset with the normalized value
df_dia

% renaming df_dia as df_dia1
df_dia1 = df_dia;

% printing df_dia1
df_dia1
%% 
% checking for missing values in dataset
sum(ismissing(df_dia1))
%% 
% checking for zero values of Glucose column
sum(~df_dia1.Glucose)

% replacing zero values of Glucose column with median of it corresponding to outcome column.  
z=(~df_dia1.Glucose);
sd=(df_dia1.Outcome==1);
m=median(df_dia1.Glucose(~z & sd));
df_dia1.Glucose(z & sd)=m;
m=median(df_dia1.Glucose(~z & ~sd));
df_dia1.Glucose(z & ~sd)=m;
df_dia1.Glucose

% checking for zero values of BloodPressure column
sum(~df_dia1.BloodPressure)

% replacing zero values of BloodPressure column with median of it corresponding to outcome column.
z=(~df_dia1.BloodPressure);
sd=(df_dia1.Outcome==1);
m=median(df_dia1.BloodPressure(~z & sd));
df_dia1.BloodPressure(z & sd)=m;
m=median(df_dia1.BloodPressure(~z & ~sd));
df_dia1.BloodPressure(z & ~sd)=m;
df_dia1.BloodPressure

% checking for zero values of SkinThickness column
sum(~df_dia1.SkinThickness)

% replacing zero values of SkinThickness column with median of it corresponding to outcome column.
z=(~df_dia1.SkinThickness);
sd=(df_dia1.Outcome==1);
m=median(df_dia1.SkinThickness(~z & sd));
df_dia1.SkinThickness(z & sd)=m;
m=median(df_dia1.SkinThickness(~z & ~sd));
df_dia1.SkinThickness(z & ~sd)=m;
df_dia1.SkinThickness

% checking for zero values of Insulin column
sum(~df_dia1.Insulin)

% replacing zero values of Insulin column with median of it corresponding to outcome column.
z=(~df_dia1.Insulin);
sd=(df_dia1.Outcome==1);
m=median(df_dia1.Insulin(~z & sd));
df_dia1.Insulin(z & sd)=m;
m=median(df_dia1.Insulin(~z & ~sd));
df_dia1.Insulin(z & ~sd)=m;
df_dia1.Insulin

% checking for zero values of BMI column
sum(~df_dia1.BMI)

% replacing zero values of BMI column with median of it corresponding to outcome column.
z=(~df_dia1.BMI);
sd=(df_dia1.Outcome==1);
m=median(df_dia1.BMI(~z & sd));
df_dia1.BMI(z & sd)=m;
m=median(df_dia1.BMI(~z & ~sd));
df_dia1.BMI(z & ~sd)=m;
df_dia1.BMI

% checking for zero values of DiabetesPedigreeFunction column
sum(~df_dia1.DiabetesPedigreeFunction)

% replacing zero values of DiabetesPedigreeFunction column with median of it corresponding to outcome column.
z=(~df_dia1.DiabetesPedigreeFunction);
sd=(df_dia1.Outcome==1);
m=median(df_dia1.DiabetesPedigreeFunction(~z & sd));
df_dia1.DiabetesPedigreeFunction(z & sd)=m;
m=median(df_dia1.DiabetesPedigreeFunction(~z & ~sd));
df_dia1.DiabetesPedigreeFunction(z & ~sd)=m;
df_dia1.DiabetesPedigreeFunction

% checking for zero values of Age column
sum(~df_dia1.Age)

% replacing zero values of Age column with median of it corresponding to outcome column.
z=(~df_dia1.Age);
sd=(df_dia1.Outcome==1);
m=median(df_dia1.Age(~z & sd));
df_dia1.Age(z & sd)=m;
m=median(df_dia1.Age(~z & ~sd));
df_dia1.Age(z & ~sd)=m;
df_dia1.Age

% printing df_dia1
df_dia1
%% 
% checking for outlier and filtering it
% reference - https://uk.mathworks.com/help/matlab/ref/isoutlier.html
isoutlier(df_dia1)

% renaming df_dia1 as df_dia2
df_dia2 = df_dia1;

% filling outliers
% reference - https://uk.mathworks.com/help/matlab/ref/filloutliers.html
df_dia2 = filloutliers(df_dia2,"previous","mean");
df_dia2
%% 
% summary of the dataset
% reference - https://uk.mathworks.com/help/stats/dataset.summary.html
summary(df_dia2)
%% 
% plotting Correlation and heatmap
% reference - https://uk.mathworks.com/help/econ/corrplot.html
% reference - https://uk.mathworks.com/help/matlab/ref/heatmap.html
corr = corrplot(df_dia2);
xvalues = {'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'};
yvalues = {'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'};
h = heatmap(xvalues,yvalues,corr);
h.Colormap = copper;
%% 
% plotting outcome
% reference - https://uk.mathworks.com/help/matlab/ref/matlab.graphics.chart.primitive.histogram.html
figure
out_counts = df_dia2.Outcome;
histogram(out_counts)
xlabel('Outcome')
ylabel('Counts')

% renaming df_dia2 as df_dia3
df_dia3 = df_dia2;
df_dia3
%% 
% model 1 - Decision tree
% reference - https://uk.mathworks.com/help/stats/decision-trees.html#bsw6p25
cv = cvpartition(size(df_dia3,1),'KFold',10)
idx = test(cv,1)

train_data = df_dia3(~idx,:);
test_data  = df_dia3(idx,:);
testing_data = test_data(:,1:end-1)

tic
rng(0);
dt = fitctree(train_data,'Outcome') 
toc

prediction_dt = predict(dt,testing_data);

% calculating accuracy of the model
accuracy_dt=((sum((prediction_dt==table2array(test_data(:,9)))))/size(test_data,1))*100;
accuracy_dt

% calculating the error
error = 100-accuracy_dt


% plotting confusion matrix
c_matrix_dt =confusionmat(table2array(test_data(:,end)),prediction_dt)
confusionchart(c_matrix_dt)

TN = c_matrix_dt(1,1)
TP = c_matrix_dt(2,2)
FN = c_matrix_dt(2,1)
FP = c_matrix_dt(1,2)

% calculating precision
precision = (TP)/(TP+FP)

% calculating recall
recall = (TP)/(TP+FN)

% calculating F-measure
F_measure = 2*(precision * recall)/(precision+recall)

% calculating sensitivity, prevalence, errorrate and correctrate
actual = dt.Y
predicted = resubPredict(dt)
classperf(actual,predicted)

% dividing into train and test
x_training = train_data(:,1:end-1);
y_training = train_data(:,end);
x_test = test_data(:,1:end-1);
y_test  = test_data(:,end);


% calculating training accuracy
prediction_train = predict(dt,x_training);
size_ytest=size(y_training,1);
sum_decision = sum((prediction_train==table2array(y_training)));
training_acc=((sum_decision)/size_ytest)*100;
training_acc

% calculating train error
train_error = 100 - training_acc

% calculating testing accuracy
prediction_test = predict(dt,x_test);
size_test1=size(y_test,1);
sum_decision = sum((prediction_test==table2array(y_test)));
testing_acc=((sum_decision)/size_test1)*100;
testing_acc

% calculating test error
test_error = 100 - testing_acc

% calculating kfoldloss
cv_dt = crossval(dt)
kf_l = kfoldLoss(cv_dt)

% renaming df_dia3 as df_dia4
df_dia4 = df_dia3;
df_dia4

%% 
%Standardization of Pregnancies column
mean_preg =  mean(df_dia4.Pregnancies);
std_preg = std(df_dia4.Pregnancies);
stan_preg = (df_dia4.Pregnancies - mean_preg)/std_preg;
df_dia4.Pregnancies = stan_preg;
df_dia4.Pregnancies

%Standardization of Glucose column
mean_glu =  mean(df_dia4.Glucose);
std_glu = std(df_dia4.Glucose);
stan_glu = (df_dia4.Glucose - mean_glu)/std_glu;
df_dia4.Glucose = stan_glu;
df_dia4.Glucose

%Standardization of BloodPressure column
mean_blo =  mean(df_dia4.BloodPressure);
std_blo = std(df_dia4.BloodPressure);
stan_blo = (df_dia4.BloodPressure - mean_blo)/std_blo;
df_dia4.BloodPressure = stan_blo;
df_dia4.BloodPressure

%Standardization of SkinThickness column
mean_ski =  mean(df_dia4.SkinThickness);
std_ski = std(df_dia4.SkinThickness);
stan_ski = (df_dia4.SkinThickness - mean_ski)/std_ski;
df_dia4.SkinThickness = stan_ski;
df_dia4.SkinThickness

%Standardization of Insulin column
mean_in =  mean(df_dia4.Insulin);
std_in = std(df_dia4.Insulin);
stan_in = (df_dia4.Insulin - mean_in)/std_in;
df_dia4.Insulin = stan_in;
df_dia4.Insulin

%Standardization of BMI column
mean_bmi =  mean(df_dia4.BMI);
std_bmi = std(df_dia4.BMI);
stan_bmi = (df_dia4.BMI - mean_bmi)/std_bmi;
df_dia4.BMI = stan_bmi;
df_dia4.BMI

%Standardization of DiabetesPedigreeFunction column
mean_dpf =  mean(df_dia4.DiabetesPedigreeFunction);
std_dpf = std(df_dia4.DiabetesPedigreeFunction);
stan_dpf = (df_dia4.DiabetesPedigreeFunction - mean_dpf)/std_dpf;
df_dia4.DiabetesPedigreeFunction = stan_dpf;
df_dia4.DiabetesPedigreeFunction

%Standardization of Age column
mean_age =  mean(df_dia4.Age);
std_age = std(df_dia4.Age);
stan_age = (df_dia4.Age - mean_age)/std_age;
df_dia4.Age = stan_age;
df_dia4.Age

df_dia4
%% 
% model 2 - Naive Bayes
% reference - https://uk.mathworks.com/help/stats/classification-naive-bayes.html
cv1 = cvpartition(size(df_dia3,1),'KFold',10)
idx1 = test(cv,1)
train_data1 = df_dia4(~idx1,:);
test_data1  = df_dia4(idx1,:);
testing_data1 = test_data1(:,1:end-1)

tic
rng(0);
nb = fitcnb(train_data1,'Outcome') 
toc

prediction_nb = predict(nb,testing_data1)

% calculating accuracy of the model
accuracy_nb=((sum((prediction_nb==table2array(test_data1(:,9)))))/size(test_data1,1))*100;
accuracy_nb

% calculating the error
error1 = 100-accuracy_nb

% plotting confusion matrix
c_matrix_nb =confusionmat(table2array(test_data1(:,end)),prediction_nb)
confusionchart(c_matrix_nb)

TN1 = c_matrix_nb(1,1)
TP1 = c_matrix_nb(2,2)
FN1 = c_matrix_nb(2,1)
FP1 = c_matrix_nb(1,2)

% calculating precision
precision1 = (TP1)/(TP1+FP1)

% calculating recall
recall1 = (TP1)/(TP1+FN1)

% calculating F-measure
F_measure1 = 2*(precision1 * recall1)/(precision1+recall1)

% calculating sensitivity, prevalence, errorrate and correctrate
actual1= nb.Y
predicted1 = resubPredict(nb)
classperf(actual1,predicted1)

% dividing into train and test
x_training1 = train_data1(:,1:end-1);
y_training1 = train_data1(:,end);
x_test1 = test_data1(:,1:end-1);
y_test1  = test_data1(:,end);

% calculating training accuracy
prediction_train1 = predict(nb,x_training1);
size_ytest2=size(y_training1,1);
sum_nb = sum((prediction_train1==table2array(y_training1)));
training_acc1=((sum_nb)/size_ytest2)*100;
training_acc1

% calculating train error
train_error1 = 100 - training_acc1

% calculating testing accuracy
prediction_test1 = predict(dt,x_test1);
size_test3=size(y_test1,1);
sum_nb = sum((prediction_test1==table2array(y_test1)));
testing_acc1=((sum_nb)/size_test3)*100;
testing_acc1

% calculating test error
test_error1 = 100 - testing_acc1

% calculating kfoldloss
cv_nb = crossval(nb)
kf_l1 = kfoldLoss(cv_nb)

%% 
% plotting ROC curve
[labels, scores] = resubPredict(dt)
[X1,Y1] = perfcurve(dt.Y,scores(:,2),1)
plot(X1,Y1,'LineWidth',1)
hold on
[labels1, scores1] = resubPredict(nb)
[X2,Y2] = perfcurve(nb.Y,scores1(:,2),1)
plot(X2,Y2,'LineWidth',1)
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC')
hold off

% Checking the value of AUC
trapz(X1,Y1)
trapz(X2,Y2)

%% 
% plotting kfold percentage splitting
a = categorical({'DT','NB'});
a = reordercats(a,{'DT','NB'});
ax1 = subplot(2,2,1);
b = [precision precision1];
bar(ax1,a,b,'FaceColor','c')
xlabel('Classification Algorithm')
ylabel('Precision')

ax2 = subplot(2,2,2);
b1 = [recall recall1];
bar(ax2,a,b1,'FaceColor','y')
xlabel('Classification Algorithm')
ylabel('Recall')

ax3 = subplot(2,2,3);
b2 = [F_measure F_measure1];
bar(ax3,a,b2,'FaceColor','k')
xlabel('Classification Algorithm')
ylabel('F-measure')

ax4 = subplot(2,2,4);
b3 = [accuracy_dt accuracy_nb];
bar(ax4,a,b3,'FaceColor',[0 .4 .4])
xlabel('Classification Algorithm')
ylabel('Accuracy')

sgtitle('Kfold percentage splitting')
