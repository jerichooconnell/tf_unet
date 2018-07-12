function [im_soft,im_hard] = net_python(yy,label, train)

if train
    x = reshape(yy,numel(yy(:,:,1)),length(yy(1,1,:)));
    t = reshape(label,numel(label));
    
    %% Training Neural Network ************************************************
    %Setting the random number generator, so that can compare results
    setdemorandstream(491218382)
    % Just loading the neural network, uncomment the next lines to train the
    % network
    %load('net_r6')
    % Initializing a neural network with 30 entries
    net = fitnet(30);%55
    %view(net)
    %Training the network with the training data
    %[net,tr] = train(net,x,t); JO
    [net,~] = train(net,x,t);
    %nntraintool
    %Plotting the performance
    %plotperform(tr)
    %     % Testing a random sample of the test data
    %     testX = x(:,tr.testInd);
    %     testT = t(:,tr.testInd);
    %     testY = net(testX);
    %     % Analyzing the performance
    %     perf = mse(net,testT,testY)
    %     y = net(x);
    %     % Looking at the regression
    %     plotregression(t,y)
    %     e = t - y;
    %     %Histogram of errors
    %     figure
    %     ploterrhist(e)
    save('net_r6','net')
else
    
    Xnew = reshape(yy,numel(yy(:,:,1)),length(yy(1,1,:)));
    load('net_r6')
    
    ys = net(Xnew');
    ys = ys';
    
    im_soft = reshape(ys(:,1),size(yy,1),size(yy,2));
    im_hard = reshape(ys(:,2),size(yy,1),size(yy,2));
end