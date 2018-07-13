function perf = net_cal(x,t)

x_train = reshape(x,4,numel(x)/4);

net = cascadeforwardnet(55); %55
net.trainParam.showWindow = 0;
[net, perf] = train(net, x_train, t); %JO

save('net','net')

end