clear; close all; clc
nlevels = 12;
nreps = 5;

load samples_chip1;
load pH;
featurenum = size(X, 1);

X = X';

for i=1:nlevels
    figure;
    plot(wav, X((i-1)*nreps+1:i*nreps,:));
    axis([-100 3000 -2000 16000]);
    xlabel('Raman Shift (cm^{-1})');
    ylabel('Raman Intensity (a.u.)');
    title(num2str(pH(i*nreps), 'pH = %.2f'));
end