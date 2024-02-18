% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by Neural Pattern Recognition app
% Created 27-Nov-2022 17:55:49
%
% This script assumes these variables are defined:
%
%   input_data - input data.
%   target_data - target data.

% Generate input data
run('gen_input_target_data2.m')

x = input_data;
t = target_data;

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% % Create a Pattern Recognition Network
% hiddenLayerSize = 10;
% net = patternnet(hiddenLayerSize, trainFcn);
% 
% % Setup Division of Data for Training, Validation, Testing
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
% 
% % Set number of epochs. Default is 1000.
% net.trainParam.epochs = 100;
% 
% % Train the Network
% [net,tr] = train(net,x,t);
% 
% % Test the Network
% y = net(x);
% e = gsubtract(t,y);
% performance = perform(net,t,y)
% % epoch
% tind = vec2ind(t);
% yind = vec2ind(y);
% percentErrors = sum(tind ~= yind)/numel(tind);
% 
% % View the Network
% view(net)

% Generate networks with hiddenlayer size 5 to 50
perf_data = 1:10;
percentErrors = 1:10;
for hiddenLayerSize = 5:5:50

    % Train the Network
    net = patternnet(hiddenLayerSize, trainFcn);         % Generate network
    net.trainParam.epochs = 100;
    net.trainParam.max_fail = 100;
    [net, tr] = train(net, x, t);        % Train network
    
    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    perf_data(hiddenLayerSize/5) = perform(net,t,y);
    % epoch
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors(hiddenLayerSize/5) = sum(tind ~= yind)/numel(tind);
%     perf_data(hiddenLayerSize/5) = tr.best_tperf;    % Log performance
end

figure
plot(5:5:50, perf_data, '-o')
title('Performance (Cross Entropy) vs # Neurons in Hidden Layer')
xlabel('Num Neurons')
ylabel('Performance')

figure
plot(5:5:50, percentErrors, '-o')
title('Percent-Error vs # Neurons in Hidden Layer')
xlabel('Num Neurons')
ylabel('Percent-Error %')

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)
