function [target_data, input_data] = gen_data(mu,sigma,num_features,num_samples)
    rng('default'); % set rng seed for reproducability
%     num_samples = 1000;
%     num_features = 3;
    
    input_data = zeros(num_features, num_samples);      %input vectors. seperated by col.
    target_data = randi(2, [1 num_samples]) - 1;        %target vectors. rand generated from uniform dist.
    target_data = [target_data; ~target_data];          %format for training
    feat_dist = zeros(num_features,2,2);                %normal distributions for each feature
%     mu = zeros(num_features,1,2);
%     sigma = zeros(num_features,num_features,2);
%     mu(:,1,1) = [1;3;4];                   %class 1 features
%     mu(:,1,2) = [4;2;5];                  %class 1 features
%     sigma(:,:,1) = [1 -.5 0;...
%                     -.5 4 0;...
%                     0 0 3];
%     sigma(:,:,2) = [2 0 0;...
%                     0 6 0;...
%                     0 0 1];
    
    class_data = zeros(num_features, num_samples,2);
    R = mvnrnd(mu(:,:,1),sigma(:,:,1),num_samples);
    class_data(:,:,1) = R';
    
    R = mvnrnd(mu(:,:,2),sigma(:,:,2),num_samples);
    class_data(:,:,2) = R';
    
    %Generate training data
    for i = 1:num_samples
        class_num = ~target_data(1,i) + 1;
        input_data(:,i) = class_data(:,i,class_num);
    end
end