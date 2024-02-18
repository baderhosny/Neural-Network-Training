gen_data;

% class_color = ["red";"blue"];
% % Histogram Comparing Normal distribution of class features
% for i = 1:num_features
%     figure
%     for k = 1:2
%         hold on
%         histogram(class_data(i,:,k),FaceColor=class_color(k))
%         legend(["class 1" "class 2"]);
%     end
%     title(['Distribution of feature ' int2str(i)]);
% end