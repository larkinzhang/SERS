function retv = trainPCR(X, pH, nlevels, nreps, wav)

cc = hsv(nlevels);

featurenum = size(X, 2);
sumup = sum(X,2);
X = bsxfun(@rdivide,X,sumup);
[~,PCAScores,~,~,EXPLAINED] = pca(X);

%% Explaination
fprintf('Plotting the explanation of PLS.\n\n');
figure('name', 'Explain');
plot(1:15, cumsum(EXPLAINED(1:15)),'o-');
xlabel('Number of Principal Components');
ylabel('Percent Variance Explained in X');

%% Evaluation
% Divide dataset into training set and testing set and evaluate the model.
indices = zeros(nlevels * nreps, 1);
for i = 1:nlevels
    indices((i - 1) * nreps + 1:i * nreps, :) = crossvalind('Kfold', nreps, nreps);
end

compcnt = 30;
v = zeros(compcnt*nreps, featurenum);
tot = zeros(compcnt, 1);
yfit = zeros(compcnt * nreps, nlevels);
minval = 1e+10;
pc = -1;

for k = 1:compcnt
    for i = 1:nreps
        test = (indices == i); train = ~test;
        Xtrain = X(train,:);
        pHtrain = pH(train);
        Xtest = X(test,:);
        pHtest = pH(test);

        [PCALoadings,PCAScores,PCAVar,TSQUARED,EXPLAINED] = pca(Xtrain);
        betaPCR = regress(pHtrain - mean(pHtrain), PCAScores(:,1:k));
        betaPCR = PCALoadings(:,1:k) * betaPCR;
               
        v((k-1)*nreps+i, :) = betaPCR';
        
        betaPCR = [mean(pHtrain) - mean(Xtrain) * betaPCR; betaPCR];
        yfitPCR = [ones(size(Xtest,1),1) Xtest] * betaPCR;

        SMSE = (sum((yfitPCR - pHtest) .^ 2) / sum((pHtest - mean(pHtest)) .^ 2));
        tot(k) = tot(k) + SMSE;
        yfit((k - 1) * nreps + i, :) = yfitPCR';
    end
    
    tot(k) = tot(k) / nreps;
    if tot(k) < minval
        minval = tot(k);
        pc = k;
    end
end

fprintf('The smallest average SMSE is %f with %d principal components.\n', minval, pc);

figure('name', 'Grid Search');
plot(1:compcnt, tot, 'o-');
xlabel('Principal Components');
ylabel('SMSE');

figure('name', 'Predict');
hold on;
for i = 1:nreps
    plot(pHtest,yfit((pc - 1) * nreps + i, :),'bo');
    xlabel('Observed Response');
    ylabel('Fitted Response');
end
lx = [min(pHtest) max(pHtest)];
ly = lx;
plot(lx, ly);
hold off;

figure('name', 'Visualization of v');
plot(wav, mean(v((pc - 1) * 5 + 1:pc * 5,:)));
axis([-100 3000 -300 300]);

[PCALoadings,PCAScores,PCAVar,TSQUARED,EXPLAINED] = pca(X);
retv = regress(pH - mean(pH), PCAScores(:,1:pc));
retv = PCALoadings(:,1:pc) * retv;
retv = [mean(pH) - mean(X) * retv; retv];

end
