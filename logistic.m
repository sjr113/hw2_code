function [w,J] = logistic(X, y)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
[P, N] = size(X);
w = rand(P+1, 1)*0.1;
au_x = vertcat(ones(1, N), X);
learning_rate = 0.1;
y(y==-1) = 0;

hypothesis = sigmoid(w' * au_x); 
% disp(size(hypothesis));
delta_J = log(hypothesis).*y + (1-y).*log(1-hypothesis);
J = -sum(delta_J)./N;

k = 0;
while k<1000000 && abs(J)>1e-4
    k = k+ 1;
    hypothesis = sigmoid(w' * au_x);  % º∆À„ ‰≥ˆ÷µ
    delta = hypothesis - y;
    w = w - (learning_rate)./N * (au_x * delta');
    delta_J = log(hypothesis).*y + (1-y).*log(1-hypothesis);
    J = -sum(delta_J, 2)./N;

end
end
function y = sigmoid(x)
    y = 1.0./(1 + exp(-x));
end
