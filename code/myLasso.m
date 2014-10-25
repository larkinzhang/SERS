
clear; close all; clc


nlevels = 12;
nreps = 5;

load samples_chip1_new;
load pH2;

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

%load lambda;
%lambda = lambda';
lambda = 0.0001:0.0001:0.0021;
%lambda=[0.0001 0.0008];
lambdacnt = size(lambda,2);
tot = zeros(lambdacnt,1);
v = zeros(lambdacnt*nreps, 1000-200+1);

fhandle = zeros(lambdacnt,1);

for i = 1:lambdacnt
    fhandle(i) = figure;
    xlabel('Observed Response');
    ylabel('Fitted Response');  
    lx = [min(pH) max(pH)];
    ly = lx;
    plot(lx, ly);
end

for i = 1:nreps
        test = (indices == i); train = ~test;
        Xtrain = Xtrim(train,:);
        pHtrain = pH(train);
        Xtest = Xtrim(test,:);
        pHtest = pH(test);

        B = lasso(Xtrain,pHtrain - mean(pHtrain),'Lambda',lambda);
   
        for k = 1:lambdacnt
            figure(fhandle(k));
            hold on;
            betaLasso = B(:,k);
            v((k-1)*nreps+i,:) = betaLasso;
        
            betaLasso = [mean(pHtrain) - mean(Xtrain) * betaLasso; betaLasso];
            yfitLasso = [ones(size(Xtest,1),1) Xtest] * betaLasso;
            plot(pHtest,yfitLasso,'bo');
            

            SMSE = (sum((yfitLasso - pHtest) .^ 2) / sum((pHtest - mean(pHtest)) .^ 2));
            tot(k) = tot(k) + SMSE;
        end
end

for k=1:lambdacnt
    fprintf('The average of SMSE with lambda = %f is %f\n', lambda(k), tot(k) / nreps);
end



%figure;
%hold on;
%plot(wav(200:1000), mean(v));
%plot(wav(1:199), zeros(1,199));
%plot(wav(1001:1044), zeros(1,44));
%xlabel('Raman Shift (cm^{-1})');
%axis([-100 3000 -0.004 0.002])
%hold off;
