function [w, num] = svm(X, y)
%SVM Support vector machine.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           num:  number of support vectors
%

% YOUR CODE HERE
% 调用matlab的函数quadprog解二次优化方程
H = (y' * y).* (X' *X); % linear kernel
[P, N] = size(X);
f = -ones(N, 1);
A =  [];
b = [];
Aeq = y;
beq = 0;
lb = zeros(N, 1);
ub = 1000* ones(N , 1);
a0 = zeros(N, 1);
options = optimset;
options.LargeScale = 'on';
options.Display = 'off';
options.Algorithm = 'active-set';
% 调用优化函数quadprog直接进行求解
[a, fval, exitflag, output, lambda] = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
% 求支持向量的下标索引
index_sv = find(a>1e-8);

temp = (y.*a') * (X'* X(:, index_sv));
b = 1./y(1, index_sv)-temp;
b = mean(b);
[num, aaaa] = size(index_sv);

% fprintf('Construct function Y = sign(tmp+b):')

w = zeros(P, 1);
for i = 1:N
    w = w + a(i, 1)*y(1, i) * X(:, i);
end
w = vertcat(b, w);
% margin = 1/norm(a);
end
