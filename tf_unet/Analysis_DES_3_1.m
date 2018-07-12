function [CNR, perf] = Analysis_DES_3_1(cal, val, evec, target_SN, layer)

%% Adding noise combined with MTF

% Script for calculating the signal to noise for the images

global verbose;
global count2;
global global_n;

%% Making the images

load('flood_3_0.mat');

corr = (10000 - Flood_3_0);

cal = cal + corr;
val = val + corr;
original = cal(:, :, 1);
ratio_ii = 1 / (length(evec) - 1);

for ii = 1:length(cal(1, 1, :))
    
    % This is the total percentage of the fluence in the image
    cal(:, :, ii) = cal(:, :, ii) .* (1 - ratio_ii * (ii - 1));
    val(:, :, ii) = val(:, :, ii) .* (1 - ratio_ii * (ii - 1));
    
    for jj = ii + 1:length(cal(1, 1, :))
        
        % This is the total percentage of the fluence in the energy bin
        % ratio_jj = integrate_spec([evec(jj) evec(jj+1)]);
        
        cal(:, :, ii) = cal(:, :, ii) - cal(:, :, jj) .* ratio_ii;
        val(:, :, ii) = val(:, :, ii) - val(:, :, jj) .* ratio_ii;
        
        
    end
end

%% Getting the signal to noise in the validation
% Using the signal to noise in water

y = 205;
x = 201;

count2 = 1;


% Finding the singnal in water that will be the basis for the noise
signal = 3.1459e+03;
% Finding the target noise
sigma = signal / target_SN;
% Finding the signal strength that would give the correct poisson noise
correct_signal = (signal / sigma)^2;

% Now we need to scale the image by the correct factor so that the sqrt
% of the image is the noise

%% Generating the thickness with the given Noise

% !! This value is hardcoded and dependant on the spectrum = sqrt(.4319) &
% sqrt(1 - .4319)

for ii = 1:length(cal(1, 1, :))
    
    val(:, :, ii) = val(:, :, ii) .* correct_signal / signal + (normrnd(0, sqrt(val(:, :, ii) ...
        .*correct_signal/signal), [300, 300]));
    
end


load('x_3_0')
load('y_3_0')

t_hard = [4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0, 0, 2, 2.5, 3, 3.5, 4, 0.5, 1, 1.5, 0, 2, 2.5, 3, 3.5, 4, 0.5, 1, 1.5, 0];    t_v = [3.9, 3.4, 2.9, 2.4, 1.9, 1.4, 0.90, 0.40, 0, 0, 1.9, 2.4, 2.9, 3.4, 3.9, 0.40, 0.90, 1.4, 0, 1.9, 2.4, 2.9, 3.4, 3.9, 0.40, 0.90, 1.4, 0];
nums = length(t_hard);

yy = zeros(size(val));
Y = zeros(length(t_hard), length(cal(1, 1, :)));

cal(cal < 0) = 0.00001;
val(val < 0) = 0.00001;

for ii = 1:length(cal(1, 1, :))
    ratio_ii = 1 / (length(evec) - 1);
    yy(:, :, ii) = log(cal(:, :, ii)/((ratio_ii) * 10000));
end

for ii = 1:length(cal(1, 1, :))
    temp_ML = returnContours_3_1(yy(:, :, ii), x, y, t_hard);
    if ii ~= 1
        Y_ML = cat(1, Y_ML, temp_ML(end, :));
    else
        Y_ML = temp_ML;
    end
end

for ii = 1:length(cal(1, 1, :))
    ratio_ii = 1 / (length(evec) - 1);
    yy(:, :, ii) = log(val(:, :, ii)/((ratio_ii) * 10000));
end

for ii = 1:length(cal(1, 1, :))
    temp_ML = returnContours_3_1(yy(:, :, ii), x, y, t_v);
    if ii ~= 1
        Y_ML_val = cat(1, Y_ML_val, temp_ML(end, :));
    else
        Y_ML_val = temp_ML;
    end
end

x_train = cat(1, Y_ML(2:end, :)); %,Y_ML(end,:)./Y_ML(3,:));
t = Y_ML(1, :);
x_val = Y_ML_val(2:end, :);
t_val = Y_ML_val(1, :);

%% Training Neural Network ************************************************
%Setting the random number generator, so that can compare results
% Just loading the neural network, uncomment the next lines to train the
% network
%load('net_r6')
% Initializing a neural network with 30 entries\
Xnew = reshape(yy, numel(yy(:, :, 1)), length(yy(1, 1, :)));
Xnew = Xnew';
%Xnew = (Xnew(1:end-1,:)./Xnew(end,:));
x2 = x(1:8);
y2 = y(1:8);

%x2 = x2(x2 ~= 0);
%y2 = y2(y2 ~= 0);

Diff = [3.9, 3.4, 2.9, 2.4, 1.9, 1.4, 0.9, 0.4, 0];

lll = zeros(1, 5);
npts = length(x2);
CNR = zeros(5, npts, 2);
%     %     lly = zeros(1,10);
%     for layer = max(min(layers)-1,1):1:max(layers)+1
layer5 = layer * 5;
for runs = 1:5
    
    net = cascadeforwardnet(layer5); %55
    net.trainParam.showWindow = 0;
    [net, ~] = train(net, x_train, t); %JO
    
    testY = net(x_val);
    perf = mse(net, t_val, testY);
    
    lll(runs) = perf;
    
    ys = net(Xnew);
    %ys = net(cat(1,Xnew,Xnew(end,:)./Xnew(end-1,:)));
    ys = ys';
    
    im_hard = reshape(ys(:, 1), size(yy, 1), size(yy, 2));
    
    [ROI, SD] = createContoursSD(im_hard, x2, y2);
    [ROI_w, SD_w] = createContoursSD(im_hard, 201, 198);
    
    
    for pp = 1:8
        
        CNR(runs, pp, 1:2) = (ROI(pp) - ROI_w) ./ sqrt(SD(pp, :).^2+mean(SD_w)^2);
    end
    
    %lly(runs,ind) = I(1);
end
%         ind = ind + 1;


perf = mean(lll);
CNR = squeeze(mean(CNR));

%% Making the validation image

if verbose > -1 && target_SN == 100
    figure(15)
    subplot(3, 5, (length(evec) - 1)+10)
    imagesc(im_hard)
    title('Cartilage SNR = 100')
    caxis([0, 8])
    axis image
    axis off
    colorbar
    colormap gray
    hold on
end
end