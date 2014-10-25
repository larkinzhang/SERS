
clear; close all; clc


nlevels = 12;
nreps = 5;

load samples_chip1s_new;
load pH2;

featurenum = size(X, 1);

X = X';
sumup = sum(X,2);
X = bsxfun(@rdivide,X,sumup);

[PCALoadings,PCAScores] = pca(X);
X = X * PCALoadings(:,1:9);

%% Evaluation
% Divide dataset into training set and testing set and evaluate the model.
load index;

sigma = 0.0021:0.0001:0.004;
sigmacnt = size(sigma,2);
%v = zeros(compcnt*nreps, featurenum);

nsq = sum(X.^2,2);
K = bsxfun(@minus,nsq,(2*X)*X.');
K = bsxfun(@plus,nsq.',K);

for k = 1:sigmacnt
    tot = 0;
    figure('name', num2str(sigma(k),'Kernel regression with sigma = %d'));
    hold on;
    
    for i = 1:nreps
        test = (indices == i); train = ~test;
        Xtrain = X(train,:);
        pHtrain = pH(train);
        Xtest = X(test,:);
        pHtest = pH(test);

        curK = exp(-K ./ (sigma(k) ^ 2));
        curK = curK(test, train);
        
        wsum = bsxfun(@times, curK, pHtrain');
        predict = sum(wsum,2) ./ sum(curK,2);
        
        plot(pHtest, predict,'bo');
        
        
        SMSE = (sum((predict - pHtest) .^ 2) / sum((pHtest - mean(pHtest)) .^ 2));
        tot = tot + SMSE;
    end
    lx = [min(pHtest) max(pHtest)];
    ly = lx;
    plot(lx, ly);
    hold off;

    fprintf('The average of SMSE with sigma = %d is %f\n', sigma(k), tot / nreps);
end
