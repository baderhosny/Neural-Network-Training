clc
clear

num_samples = 1000;
num_features = 3;

mu = zeros(num_features,1,2);
sigma = zeros(num_features,num_features,2);
mu(:,1,1) = [1;3;4];                   %class 1 features
mu(:,1,2) = [4;2;5];                  %class 1 features
sigma(:,:,1) = [1 0 0;...
                0 4 0;...
                0 0 3];
sigma(:,:,2) = [2 0 0;...
                0 6 0;...
                0 0 1];

[x,t] = gen_data(mu,sigma,num_features,num_samples);

trainFcn = 'traingda';  % Scaled conjugate gradient backpropagation.
% Generate networks with hiddenlayer size 5 to 50


hiddenLayerSize = 1;
count = 1;

lr_range = linspace(0.1,0.5,20);

perf_data = 1:length(lr_range);
percentErrors = 1:length(lr_range);

for  learning_rate = lr_range

    % Train the Network
    net = patternnet(repmat(25,1,hiddenLayerSize), trainFcn);         % Generate network
    net.trainParam.lr = learning_rate;
    net.trainParam.epochs = 100;
    net.trainParam.max_fail = 100;
    net.trainParam.showWindow = false;
    [net, tr] = train(net, x, t);        % Train network
    
    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    perf_data(count) = perform(net,t,y);
    % epoch
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors(count) = sum(tind ~= yind)/numel(tind);
%     perf_data(hiddenLayerSize) = tr.best_tperf;    % Log performance
    count = count + 1;
end

figure
plot(lr_range, perf_data,'-o')
title('Performance (Cross Entropy) vs Learning Rate')
xlabel('Learning Rate')
ylabel('Performance')

figure
plot(lr_range, percentErrors,'-o')
title('Percent-Error vs Learning Rate')
xlabel('Learning Rate')
ylabel('Percent-Error %')
