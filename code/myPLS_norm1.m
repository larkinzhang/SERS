clear; close all; clc

nlevels = 12;
nreps = 5;

%% Virtualization
fprintf('Visualizing dataset for PLS.\n\n');
cc = hsv(20);
load samples_chip1_new;
load pH2;

featurenum = size(X, 1);

X = X';
sumup = sum(X,2);
X = bsxfun(@rdivide,X,sumup);
[n,m] = size(X);
[~,~,Xscores,~,~,PLSPctVar] = plsregress(X,pH);

figure('name', 'PLS');
hold on;
pHindex = zeros(nlevels, 1);
for i = 0:nlevels - 1
    plot(Xscores(i*nreps+1:i*nreps+nreps,1), Xscores(i*nreps+1:i*nreps+nreps,2), 'o-', 'color', cc(i+1,:));
    pHindex(i + 1) = i * nreps + 1;
end

legendCell = cellstr(num2str(pH(pHindex), '%.2f'));
legend(legendCell, 'Location', 'BestOutside');
hold off;

figure('name', 'PLS');
hold on;
for i = 0:nlevels - 1
    plot(mean(Xscores(i*nreps+1:i*nreps+nreps,1)), mean(Xscores(i*nreps+1:i*nreps+nreps,2)), 's', 'color', cc(i+1,:));
end
legend(legendCell, 'Location', 'BestOutside');
hold off;

fprintf('Program paused. Press enter to continue.\n\n');
pause;

%% Explaination
fprintf('Plotting the explanation of PLS.\n\n');
figure('name', 'Explain');
plot(1:15, 100*cumsum(PLSPctVar(1,1:15)),'o-');
xlabel('Number of Principal Components');
ylabel('Percent Variance Explained in X');
fprintf('Program paused. Press enter to continue.\n\n');
pause;

%% Evaluation
% Divide dataset into training set and testing set and evaluate the model.
fprintf('Divide dataset into train set and test set and evaluate the model.\n\n');
load index;

compcnt = 15;
v = zeros(compcnt*nreps, featurenum);

for k = 1:compcnt
    tot = 0;
    
    figure('name', num2str(2*k, 'PLSR with %d Components'));
    hold on;

	for i = 1:nreps
        test = (indices == i); train = ~test;
        Xtrain = X(train,:);
        pHtrain = pH(train);
        Xtest = X(test,:);
        pHtest = pH(test);
   
        [Xloadings,Yloadings,Xscores,Yscores,betaPLS,PLSPctVar] = plsregress(Xtrain,pHtrain,2*k);
        
        v((k-1)*nreps+i, :) = betaPLS(2:featurenum+1,:)';
        
        yfitPLS = [ones(size(Xtest,1),1) Xtest]*betaPLS;
        plot(pHtest,yfitPLS,'bo');
        xlabel('Observed Response');
        ylabel('Fitted Response');
    
        SMSE = (sum((yfitPLS - pHtest) .^ 2) / sum((pHtest - mean(pHtest)) .^ 2)) / nlevels;
        tot = tot + SMSE;
    end
    lx = [min(pHtest) max(pHtest)];
    ly = lx;
    plot(lx, ly);
    hold off;

    fprintf('The average of SMSE with %d components is %f\n', 2*k, tot / nreps);
end

figure('name', 'figure for v');
[PCALoadings,PCAScores,PCAVar] = pca(v);
hold on;
for i = 0:compcnt - 1
    plot(PCAScores(i*nreps+1:i*nreps+nreps,1), PCAScores(i*nreps+1:i*nreps+nreps,2), 's-', 'color', cc(i+1,:));
end

legendCell = cellstr(num2str(((1:5)*2)', '%d components'));
legend(legendCell, 'Location', 'BestOutside');
hold off;