function perf = net_cal(x,t)

%save('x','x')
%save('t','t')

x_train = reshape(x,numel(x)/4,4);

%save('x_train','x_train')

net = cascadeforwardnet(25); %55
net.trainParam.showWindow = 0;
[net, perf] = train(net, x_train', t); %JO

save('net','net')

end