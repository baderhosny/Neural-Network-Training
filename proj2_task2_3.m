% Two layer network

clc;    % clear all vars and console
clear;

x = -12:.1:12;                       % Function input from -12 to 12
t = 100 + cos(pi .* x ./ 12) + sin(pi .* x ./ 5);           % Training output.   % Target Function output

% Network performance (as Mean Squared Error). Closer to zero = better
lr_range = linspace(0.1,0.5,20);
perf_data = 1:length(lr_range);
percentErrors = 1:length(lr_range);          
trainFcn = 'trainlm';                    % Set Training funct to 
                                        % 'Levenberg-Marquardt' because
                                        % it is fastest
count = 1;
hiddenLayerSize = 25;
% Generate networks with hiddenlayer size 5 to 50
for learning_rate = lr_range
    net = fitnet(hiddenLayerSize, trainFcn);         % Generate network
    net.trainParam.lr = learning_rate;
    net.trainParam.epochs = 100;
    net.trainParam.max_fail = 100;
    net.trainParam.showWindow = false;
    [net, tr] = train(net, x, t);        % Train network
%     perf_data(count) = tr.best_tperf;    % Log performance

    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    perf_data(count) = perform(net,t,y);
    % epoch
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors(count) = sum(tind ~= yind)/numel(tind);
    count = count + 1;
end

figure
plot(lr_range, perf_data,'-o')
title('Performance (Mean-Square Error) vs Learning Rate')
xlabel('Learning Rate')
ylabel('Mean-Square Error')

figure
plot(lr_range, percentErrors,'-o')
title('Percent-Error vs Learning Rate')
xlabel('Learning Rate')
ylabel('Percent-Error %')