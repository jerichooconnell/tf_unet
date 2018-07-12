function [im_hard] = Analysis_DES_validation(val,nbins,target_SN,width,height)

diary on
val = reshape(val,width,height,nbins);

%% The signal is hard coded

% Finding the singnal in water that will be the basis for the noise
signal = 3.1459*10^3;
% Finding the target noise
sigma = signal / target_SN;
% Finding the signal strength that would give the correct poisson noise
correct_signal = (signal / sigma)^2;

% Now we need to scale the image by the correct factor so that the sqrt
% of the image is the noise

%% Generating the thickness with the given Noise
ll = length(val(:,1,1));

for ii = 1:length(val(1, 1, :))
    
    val(:, :, ii) = val(:, :, ii) .* correct_signal ./ signal + normrnd(0,sqrt(val(:, :, ii) ...
       .*correct_signal./signal), [ll, ll]);
    
end

size(val)
% Reshaping
yy = zeros(size(val));

val(val < 0) = 0.00001;

for ii = 1:length(val(1, 1, :))
    ratio_ii = 1. / double(nbins);
    yy(:, :, ii) = log(double(val(:, :, ii))/double((ratio_ii) * 10000));
end

%% Vaidating Neural Network ************************************************

load('net.mat')

Xnew = reshape(yy, numel(yy(:, :, 1)), length(yy(1, 1, :)));
Xnew = Xnew';

ys = net(Xnew);

ys = ys';

im_hard = reshape(ys(:, 1), size(yy, 1), size(yy, 2));

end