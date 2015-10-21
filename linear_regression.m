function w = linear_regression(X, y)
%LINEAR_REGRESSION Linear Regression.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
% ������xת��Ϊ����������������
[P, N] = size(X);
au_x = vertcat(ones(1, N), X);
lamda = 0.001;% ����Ȩ��˥��ϵ��

% ��ʼ��W
w = zeros(P+1, 1);
pre_y = w' * au_x;
J = sum((y - pre_y)* (y - pre_y)') * 1.0/N;
theta = 0.1;
J1 = ((y - pre_y)* (y - pre_y)') + lamda *( w'*w);
% w = inv(au_x * au_x') * au_x * y';
% ʹ�ö��׷�����Ȩ��˥����
w = inv(au_x * au_x' + lamda*ones(P+1, P+1)) * au_x * y';

end
