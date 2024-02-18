clear   % Clear previous variables
clc

rng('default'); % set rng seed for reproducability
num_samples = 1000;
num_features = 3;

input_data = zeros(num_features, num_samples);      %input vectors. seperated by col.
target_data = randi(2, [1 num_samples]) - 1;        %target vectors. rand generated from uniform dist.
target_data = [target_data; ~target_data];          %format for training
feat_dist = zeros(num_features,2,2);                %normal distributions for each feature
feat_dist(:,:,1) = [1 5;2 4;3 3];                   %class 1 features
feat_dist(:,:,2) = [3 3;4 2;5 1];                 %class 2 features

class_data = zeros(num_features, num_samples,2);

%Generate training data
for i = 1:num_samples
    for j = 1:num_features
        for k = 1:2
    %         class_num = ~target_data(1,i) + 1;
            class_num = k;
            mu = feat_dist(j,1, class_num);
            std = feat_dist(j,2, class_num);
    %             input_data(j,i) = std * randn(1) + mu;
            class_data(j,i,k) = std * randn(1) + mu;
        end
    end
    class_num = ~target_data(1,i) + 1;
    input_data(:,i) = class_data(:,i,class_num);
end

% class_color = ["red";"blue"]
% % Histogram Comparing Normal distribution of class features
% for i = 1:num_features
%     figure
%     for k = 1:2
%         hold on
%         histogram(class_data(i,:,k),FaceColor=class_color(k))
%         legend(["class 1" "class 2"])
%     end
%     title(['Distribution of feature ' int2str(i)]);
% end