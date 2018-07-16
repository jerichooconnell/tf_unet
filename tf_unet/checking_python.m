figure
subplot(221)
imagesc(image(:,:,1))
subplot(222)
imagesc(image(:,:,2))
subplot(223)
imagesc(image(:,:,3))
subplot(224)
imagesc(image(:,:,4))

x = reshape(image,1,numel(image));

load('net')

xnew = reshape(x,4,numel(x)./4);

ys = net(xnew);

im_cart = reshape(ys,512,512);

imagesc(im_cart)

%% Checking the calibration

load('x')
load('t')

x_train = reshape(x,numel(x)/4);

net = cascadeforwardnet(10); %55
[net, perf] = train(net, x_train, t); %JO

%% REshaping

x2 = reshape(matlab_reshape,4,15000);
