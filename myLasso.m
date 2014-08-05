
clear; close all; clc


nlevels = 12;
nreps = 5;

load samples_chip1;
load pH;

featurenum = size(X, 1);

X = X';

Xtrim = X(:,200:1000);

% [B,FitInfo] = lasso(Xtrim,pH);

% look at
% http://www.mathworks.co.uk/help/stats/lasso.html#bs25w54-6
% to see input/output params.

% 
% disp('Number of non-zero coeffs for each lambda value');
% FitInfo.DF

% select say index 31 (with DF = 19)
% plot(200:1000,B(:,31))

load index;

%figure('name', 'Lasso');
%hold on;

v = zeros(nreps, 1000-200+1);
load lambda;
lambda = lambda';
%lambda = 0.0001:0.0001:0.32;
%lambda=[0.0001 0.0008];
lambdacnt = size(lambda,2);
tot = zeros(lambdacnt,1);

for i = 1:nreps
        test = (indices == i); train = ~test;
        Xtrain = Xtrim(train,:);
        pHtrain = pH(train);
        Xtest = Xtrim(test,:);
        pHtest = pH(test);

        B = ridge(pHtrain,Xtrain,lambda);
        
        for k = 1:lambdacnt
            betaLasso = B(:,k);
            %v(i,:) = betaLasso;
        
            betaLasso = [mean(pHtrain) - mean(Xtrain) * betaLasso; betaLasso];
            yfitLasso = [ones(size(Xtest,1),1) Xtest] * betaLasso;
            %plot(pHtest,yfitLasso,'bo');
            %xlabel('Observed Response');
            %ylabel('Fitted Response');

            SMSE = (sum((yfitLasso - pHtest) .^ 2) / sum((pHtest - mean(pHtest)) .^ 2)) / nlevels;
            tot(k) = tot(k) + SMSE;
        end
end

%lx = [min(pHtest) max(pHtest)];
%ly = lx;
%plot(lx, ly);
%hold off;

%figure;
%hold on;
%plot(wav(200:1000), mean(v));
%plot(wav(1:199), zeros(1,199));
%plot(wav(1001:1044), zeros(1,44));
%xlabel('Raman Shift (cm^{-1})');
%axis([-100 3000 -0.004 0.002])
%hold off;
