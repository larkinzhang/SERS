clear; close all; clc


nlevels = 12;
nreps = 5;

%% Virtualization
fprintf('Visualizing dataset for PCA.\n\n');
cc = hsv(nlevels);
load samples_chip1_new;
load pH2;

featurenum = size(X, 1);

X = X';
sumup = sum(X,2);
X = bsxfun(@rdivide,X,sumup);
[~,PCAScores,~,~,EXPLAINED] = pca(X);
[n,m] = size(X);

figure('name', 'PCA');
hold on;
pHindex = zeros(nlevels, 1);
for i = 0:nlevels - 1
    plot(PCAScores(i*nreps+1:i*nreps+nreps,1), PCAScores(i*nreps+1:i*nreps+nreps,2), 'o-', 'color', cc(i+1,:));
    pHindex(i + 1) = i * nreps + 1;
end

legendCell = cellstr(num2str(pH(pHindex), '%.2f'));
legend(legendCell, 'Location', 'BestOutside');
hold off;

figure('name', 'PCA');
hold on;
for i = 0:nlevels - 1
    plot(mean(PCAScores(i*nreps+1:i*nreps+nreps,1)), mean(PCAScores(i*nreps+1:i*nreps+nreps,2)), 's', 'color', cc(i+1,:));
end

legend(legendCell, 'Location', 'BestOutside');
hold off;

fprintf('Program paused. Press enter to continue.\n\n');
pause;


%% Explaination
fprintf('Plotting the explanation of PLS.\n\n');
figure('name', 'Explain');
plot(1:15, cumsum(EXPLAINED(1:15)),'o-');
xlabel('Number of Principal Components');
ylabel('Percent Variance Explained in X');
fprintf('Program paused. Press enter to continue.\n\n');
pause;

%% Evaluation
% Divide dataset into training set and testing set and evaluate the model.
load index;

compcnt = 30;
v = zeros(compcnt*nreps, featurenum);
tot = zeros(compcnt, 1);
tot_MAE = zeros(compcnt, 1);

for k = 1:compcnt
    figure('name', num2str(k, 'PCR with %d Components'));
    hold on;
    
    
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
        plot(pHtest,yfitPCR,'bo');
        xlabel('Observed Response');
        ylabel('Fitted Response');

        SMSE = (sum((yfitPCR - pHtest) .^ 2) / sum((pHtest - mean(pHtest)) .^ 2));
        MAE = sum(abs(yfitPCR - pHtest)) / nlevels;
        tot(k) = tot(k) + SMSE;
        tot_MAE(k) = tot_MAE(k) + MAE;
    end
    lx = [min(pHtest) max(pHtest)];
    ly = lx;
    plot(lx, ly);
    hold off;

    fprintf('The average of SMSE and MAE with %d components are %f and %f\n', k, tot(k) / nreps, tot_MAE(k) / nreps);
end