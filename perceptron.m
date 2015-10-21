function [w, iter] = perceptron(X, y)
%PERCEPTRON Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE
[P, N] = size(X);
% 初始化W
% w = rand(P+1, 1);
%w = zeros(P+1, 1);
w = rand(P+1, 1)* 0.01;
ying = 0.001*ones(10000,1);  % 初始化学习率
%----------------------------------------------------------------
%% 使用‘pattern classification’书里包含'normalization'的方法
% 调整输入X，将输入样本中类别为-1的全部取负
% 这一步处理在‘pattern classification’书里被称为'normalization'
% for i =1:N
%     if y(1, i)<0
%         X(:, i) = - X(:, i);
%     end
% end
% au_x = vertcat(ones(1, N), X);
% 此时增广特征向量au_x的大小为P+1 * N
% 
% 预测输出，大小为1*N
% Batch perceptron
% theta = ones(10000,1)*(1e-6);
% k = 0;
% w = rand(P+1, 1)* 0.01;
% pre_y = w' * au_x;
% 
% 返回被错误分类的样本的索引
% index_misclassified = find(pre_y<=0);
% while  ying(k) * sum(X(:, index_misclassified), 2)>theta
% while ~isempty(index_misclassified)
%     disp('index_misclassified');
%     disp(index_misclassified);
%     k = k+1;
%     w = w + ying(k) * sum(au_x(:, index_misclassified), 2);  % sum大小为P+1 * 1
%     pre_y = w' * au_x;
%     index_misclassified = find(pre_y<=0); 
%     disp('pre_y')
%     disp(pre_y);
% end

%----------------------------------------------------------------
% %% 使用输出符号直接判别类别的方法
au_x = vertcat(ones(1, N), X);

pre_y = w' * au_x; 
pre_y = sign(pre_y); % 注意符号函数sign里面输出结果是有0的
% pre_y(find(pre_y==0)) = 1;  % 将sign输出结果中为0的数置为1
pre_y(pre_y<0) = -1;
pre_y(pre_y>=0) =1;
% 被错误分类的样本的索引
index_misclassified = find(pre_y~=y);
[aaa, wrong_num] = size(index_misclassified);
k = 0;
while wrong_num~=0 && k<10000
    k = k+1;
    yy = repmat(y(1, index_misclassified), P+1, 1);
    w = w + ying(k) * sum(au_x(:, index_misclassified) .* yy, 2);
    pre_y = w' * au_x;
    pre_y(pre_y<0) = -1;
    pre_y(pre_y>=0) =1;
    index_misclassified = find(pre_y~=y);
    [aaa, wrong_num] = size(index_misclassified);
end

% 得到最后收敛时的迭代次数
iter = k;


end
 
