function retpH = predictGP(X, pH, hyp)

sumup = sum(X,2);
X = bsxfun(@rdivide,X,sumup);

meanfunc = {@meanSum, {@meanLinear, @meanConst}};
covfunc = {@covMaterniso, 5};
likfunc = @likGauss;
[retpH s2] = gp(hyp, @infVB, meanfunc, covfunc, likfunc, hyp.X, hyp.pH, X);   
     
SMSE = (sum((retpH - pH) .^ 2) / sum((pH - mean(pH)) .^ 2));
MAE = sum(abs(retpH - pH)) / size(pH, 1);
fprintf('SMSE = %f\n', SMSE);
fprintf('MAE = %f\n', MAE);
fprintf('R^2 = %f\n', 1 - SMSE);