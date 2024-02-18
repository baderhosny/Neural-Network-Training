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

trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
% Generate networks with hiddenlayer size 5 to 50
perf_data = 1:2;
percentErrors = 1:2;

for hiddenLayerSize = 1

    % Train the Network
    net = patternnet(repmat(25,1,hiddenLayerSize), trainFcn);         % Generate network
    net.trainParam.epochs = 100;
    net.trainParam.max_fail = 100;
    [net, tr] = train(net, x, t);        % Train network
    
    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    perf_data(hiddenLayerSize) = perform(net,t,y);
    % epoch
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors(hiddenLayerSize) = sum(tind ~= yind)/numel(tind);
%     perf_data(hiddenLayerSize) = tr.best_tperf;    % Log performance
end

num_features = 4;

mu = zeros(num_features,1,2);
sigma = zeros(num_features,num_features,2);
mu(:,1,1) = [1;3;4;6];                   %class 1 features
mu(:,1,2) = [4;2;5;2];                  %class 1 features
sigma(:,:,1) = [1 0 0 0;...
                0 4 0 0;...
                0 0 3 0;...
                0 0 0 2];
sigma(:,:,2) = [2 0 0 0;...
                0 6 0 0;...
                0 0 1 0;...
                0 0 0 3];

[x,t] = gen_data(mu,sigma,num_features,num_samples);

for hiddenLayerSize = 2

    % Train the Network
    net = patternnet(repmat(25,1,hiddenLayerSize), trainFcn);         % Generate network
    net.trainParam.epochs = 200;
    net.trainParam.max_fail = 100;
    [net, tr] = train(net, x, t);        % Train network
    
    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    perf_data(hiddenLayerSize) = perform(net,t,y);
    % epoch
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors(hiddenLayerSize) = sum(tind ~= yind)/numel(tind);
%     perf_data(hiddenLayerSize) = tr.best_tperf;    % Log performance
end

figure
bar(3:4, perf_data)
title('Performance (Cross Entropy) vs Number of Features')
xlabel('Features')
ylabel('Performance')

figure
bar(3:4, percentErrors)
title('Percent-Error vs Number of Features')
xlabel('Features')
ylabel('Percent-Error %')

