% hyperparameter optimization of decision tree model
rng(0);
decisionmodel = fitctree(train_data,'Outcome','OptimizeHyperparameters','auto')
prediction_decisionmodel = predict(decisionmodel,testing_data)

% check if accuracy is improved or not
acc = ((sum((prediction_decisionmodel==table2array(test_data(:,9)))))/size(test_data,1))*100;
acc

% calculating the error
error_hyper = 100 - acc

% calculating resubloss
resub_loss = resubLoss(decisionmodel)
%% 
% hyperparameter optimization of naive bayes model
rng(0);
naivemodel = fitcnb(train_data1,'Outcome','OptimizeHyperparameters','auto')
prediction_decisionmodel1 = predict(naivemodel,testing_data1)

% check if accuracy is improved or not
acc1 = ((sum((prediction_decisionmodel1==table2array(test_data1(:,9)))))/size(test_data1,1))*100;
acc1

% calculating the error
error_hyper1 = 100 - acc1

% calculating resubloss
resub_loss1 = resubLoss(naivemodel)
