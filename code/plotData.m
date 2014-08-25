clear; close all; clc
nlevels = 12;
nreps = 5;

load samples_chip1_new;
load pH2;
featurenum = size(X, 1);

X = X';
%sumup = sum(X,2);
%X = bsxfun(@rdivide,X,sumup);

for i=1:nlevels
    figure;
    plot(wav, X((i-1)*nreps+1:i*nreps,:));
    legend(['1'; '2'; '3'; '4'; '5']);
%    axis([-100 3000 -0.002 0.01]);
    axis([-100 3000 -1500 30000]);
    xlabel('Raman Shift (cm^{-1})');
    ylabel('Raman Intensity (a.u.)');
    title(num2str(pH(i*nreps), 'pH = %.2f'));
end