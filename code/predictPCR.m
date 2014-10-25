function retpH = predictPCR(X, pH, v)

sumup = sum(X,2);
X = bsxfun(@rdivide,X,sumup);

retpH = [ones(size(X,1),1) X] * v;
SMSE = (sum((retpH - pH) .^ 2) / sum((pH - mean(pH)) .^ 2));
MAE = sum(abs(retpH - pH)) / size(pH, 1);
fprintf('SMSE = %f\n', SMSE);
fprintf('MAE = %f\n', MAE);
fprintf('R^2 = %f\n', 1 - SMSE);

end