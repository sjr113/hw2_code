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
% ��ʼ��W
% w = rand(P+1, 1);
%w = zeros(P+1, 1);
w = rand(P+1, 1)* 0.01;
ying = 0.001*ones(10000,1);  % ��ʼ��ѧϰ��
%----------------------------------------------------------------
%% ʹ�á�pattern classification���������'normalization'�ķ���
% ��������X�����������������Ϊ-1��ȫ��ȡ��
% ��һ�������ڡ�pattern classification�����ﱻ��Ϊ'normalization'
% for i =1:N
%     if y(1, i)<0
%         X(:, i) = - X(:, i);
%     end
% end
% au_x = vertcat(ones(1, N), X);
% ��ʱ������������au_x�Ĵ�СΪP+1 * N
% 
% Ԥ���������СΪ1*N
% Batch perceptron
% theta = ones(10000,1)*(1e-6);
% k = 0;
% w = rand(P+1, 1)* 0.01;
% pre_y = w' * au_x;
% 
% ���ر�������������������
% index_misclassified = find(pre_y<=0);
% while  ying(k) * sum(X(:, index_misclassified), 2)>theta
% while ~isempty(index_misclassified)
%     disp('index_misclassified');
%     disp(index_misclassified);
%     k = k+1;
%     w = w + ying(k) * sum(au_x(:, index_misclassified), 2);  % sum��СΪP+1 * 1
%     pre_y = w' * au_x;
%     index_misclassified = find(pre_y<=0); 
%     disp('pre_y')
%     disp(pre_y);
% end

%----------------------------------------------------------------
% %% ʹ���������ֱ���б����ķ���
au_x = vertcat(ones(1, N), X);

pre_y = w' * au_x; 
pre_y = sign(pre_y); % ע����ź���sign��������������0��
% pre_y(find(pre_y==0)) = 1;  % ��sign��������Ϊ0������Ϊ1
pre_y(pre_y<0) = -1;
pre_y(pre_y>=0) =1;
% ��������������������
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

% �õ��������ʱ�ĵ�������
iter = k;


end
 
