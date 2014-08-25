clear; close all; clc

nlevels = 12;
nreps = 5;

load samples_chip1_new;
load pH2;

featurenum = size(X, 1);

X = X';
sumup = sum(X,2);
X = bsxfun(@rdivide,X,sumup);
[n,m] = size(X);

%% Evaluation
% Divide dataset into training set and testing set and evaluate the model.
fprintf('Divide dataset into train set and test set and evaluate the model.\n\n');
load index;


tot = 0;
    
%figure('name', 'PLSR with %d Components');
hold on;

for i = 1:nreps
        test = (indices == i); train = ~test;
        Xtrain = X(train,:);
        pHtrain = pH(train);
        Xtest = X(test,:);
        pHtest = pH(test);
   
        hyp.cov = [0; 0]; hyp.mean = zeros(featurenum + 1, 1); hyp.lik = log(0.1);
        meanfunc = {@meanSum, {@meanLinear, @meanConst}};
        covfunc = {@covMaterniso, 5};
        likfunc = @likGauss;
        
        hyp = minimize(hyp, @gp, -100, @infVB, meanfunc, covfunc, likfunc, Xtrain, pHtrain);
        [pHfit s2] = gp(hyp, @infVB, meanfunc, covfunc, likfunc, Xtrain, pHtrain, Xtest);
        
        plot(pHtest,pHfit,'bo');
        xlabel('Observed Response');
        ylabel('Fitted Response');
    
        SMSE = (sum((pHfit - pHtest) .^ 2) / sum((pHtest - mean(pHtest)) .^ 2));
        tot = tot + SMSE;
end
lx = [min(pHtest) max(pHtest)];
ly = lx;
plot(lx, ly);
hold off;

fprintf('The average of SMSE is %f\n', tot / nreps);