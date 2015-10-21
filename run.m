% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.



range = [-1, 1];
dim = 2;
% 跑1000次取平均结果
large_num= 1000;



% Part1: Preceptron
nRep = 1000; % number of replicates
nTrain = 10; % number of training data
iter_sum = 0;
wrong_num = 0;
wrong_num_sum = 0;
w_g_sum = 0;
E_test_sum = 0;

for i = 1:nRep
    [XX, yy, w_f] = mkdata(large_num);
    X = XX(:, 1:nTrain);
    y = yy(1, 1:nTrain);
    [w_g, iter] = perceptron(X, y);
    w_g_sum = w_g_sum + w_g;
    % Compute training
    au_x = vertcat(ones(1, nTrain), X);
    pre_y = w_g' * au_x; 
    pre_y = sign(pre_y); % 注意符号函数sign里面输出结果是有0的
    pre_y(pre_y==0) = 1;  % 将sign输出结果中为0的数置为1
    [aaa, wrong_num] = size(find(y~=pre_y));
    wrong_num_sum = wrong_num_sum + wrong_num;
    % Sum up number of iterations
    iter_sum = iter_sum + iter;
    
    % compute test error
    t_X = XX(:, (nTrain+1):large_num);
    t_y = yy(1, (nTrain+1):large_num);
    t_au_x = vertcat(ones(1, large_num-nTrain), t_X);
    pre_y_t = w_g' * t_au_x; 
    pre_y_t = sign(pre_y_t);
    [aaa, E_test] = size(find(t_y~=pre_y_t));
    E_test_sum = E_test_sum + E_test;
end

w_g_avg = w_g_sum*1.0/nRep;

E_train = wrong_num_sum*1.0/nRep/nTrain;
E_test = E_test_sum/(large_num-nTrain)/nRep;
avgIter = iter_sum/nRep;
disp('------------perceptron--------------------------------------------');
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('Average number of iterations is %d.\n', avgIter);
plotdata(X, y, w_f, w_g, 'Pecertron');

%% Part2: Preceptron: Non-linearly separable case
nTrain = 100; % number of training data
[XN, yN, w_fn] = mkdata(large_num, 'noisy');
X = XN(:, 1:nTrain);
y = yN(1, 1:nTrain);
[w_g1, iter] = perceptron(X, y);

t_Xn = XN(:, (nTrain+1):large_num);
t_yn = yN(1, (nTrain+1):large_num);
t_au_xn = vertcat(ones(1, large_num-nTrain), t_Xn);
    
pre_y = w_g1' * t_au_xn; 
pre_y = sign(pre_y);
pre_y(pre_y==0) = 1;
[aaa, E_test] = size(find(t_yn~=pre_y));
E_test = E_test./(large_num-nTrain);
disp('------------perceptron with Non-linearly separable case------------');
fprintf('E_test of Preceptron with Non-linearly separable case is %f.\n',  E_test);
plotdata(X, y, w_fn, w_g1, 'Pecertron');
% % 
%% Part3: Linear Regression
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
w_g_sum = 0;
E_train_sum = 0;
E_test_sum = 0;
for i = 1:nRep
    [XL, yL, w_fL] = mkdata(large_num);
    X = XL(:, 1:nTrain);
    y = yL(1, 1:nTrain);
    w_g = linear_regression(X, y);
    w_g_sum = w_g_sum + w_g;
    % Compute training error
    au_x = vertcat(ones(1, nTrain), X);
    pre_y = w_g' * au_x; 
    pre_y = sign(pre_y);
    [aaa, wrong_num] = size(find(y~=pre_y));
    E_train_sum = E_train_sum + wrong_num;
    
    % compute test error
    t_Xl = XL(:, (nTrain + 1):large_num);
    t_yl = yL(1, (nTrain + 1):large_num);
    au_x_t = vertcat(ones(1, (large_num-nTrain)), t_Xl);
    pre_y_tl = w_g' * au_x_t;
    pre_y_tl = sign(pre_y_tl);
    [aaa, wrong_num] = size(find(t_yl~=pre_y_tl));
    E_test_sum = E_test_sum + wrong_num;
end
w_g_average = w_g_sum ./nRep;
E_train = E_train_sum/nRep/nTrain;
E_test = E_test_sum/nRep/(large_num-nTrain);
disp('------------Linear Regression-------------------------------------');
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_fL, w_g, 'Linear Regression');

%% Part4: Linear Regression: noisy
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
w_g_sum = 0;
E_train_sum = 0;
E_test_sum = 0;
for i = 1:nRep
    [Xln, yln, w_fln] = mkdata(large_num, 'noisy');
    X = Xln(:, 1:nTrain);
    y = yln(1, 1:nTrain);
    w_g = linear_regression(X, y);
    au_x = vertcat(ones(1, nTrain), X);
    pre_y = w_g' * au_x; 
    w_g_sum = w_g_sum + w_g;
    % compute train error
    pre_y = sign(pre_y);
    [aaa, wrong_num] = size(find(y~=pre_y));
    E_train_sum =  E_train_sum + wrong_num;
    
    % Compute testing error
    t_Xln = Xln(:, (nTrain + 1):large_num);
    t_yln = yln(1, (nTrain + 1):large_num);
    au_x_tn = vertcat(ones(1, large_num-nTrain), t_Xln);
    pre_y_tln = w_g' * au_x_tn;
    pre_y_tln = sign(pre_y_tln);
    [aaa, wrong_num] = size(find(t_yln~=pre_y_tln));
    E_test_sum = E_test_sum + wrong_num;
end
w_g_average = w_g_sum*1.0/nRep;

E_train = E_train_sum /nRep/nTrain;
E_test = E_test_sum/nRep/(large_num-nTrain);
disp('------------Linear Regression with noise--------------------------');
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_fln, w_g, 'Linear Regression: noisy');
% 
%% Part5: Linear Regression: poly_fit
load('poly_train', 'X', 'y');
load('poly_test', 'X_test', 'y_test');

w = linear_regression(X, y);
% 得到预测输出
[x1_size, x2_size] = size(X);
au_x = vertcat(ones(1, x2_size), X);
pre_y = w' * au_x;
% train_error = (y - pre_y) * (y - pre_y)';
pre_y = sign(pre_y);
[aaa, wrong_num] = size(find(y~=pre_y));
E_train = wrong_num/x2_size;

[xt1_size, xt2_size] = size(X_test);
au_x_test = vertcat(ones(1, xt2_size), X_test);
pre_y_test = w' * au_x_test;
% test_error = (y_test - pre_y_test)* (y_test - pre_y_test)';
pre_y_test = sign(pre_y_test);
[aaa, test_error] = size(find(y_test~=pre_y_test));
E_test = test_error/xt2_size;
% Compute training, testing error
disp('------------Linear Regression with poly_fit-----------------------');
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);

% poly_fit with transform
X_t = X; % CHANGE THIS LINE TO DO TRANSFORMATION
X_test_t = X_test; % CHANGE THIS LINE TO DO TRANSFORMATION
% 进行数据转换，利用hw2给出的公式进行数据的变换，将三维（二维）的数据转换为六维
[x1_size, x2_size] = size(X);
X = vertcat(ones(1, x2_size), X);
X_t(1, :) = X(1, :);
X_t(2, :) = X(2, :);
X_t(3, :) = X(3, :);
X_t(4, :) = X(2, :).*X(3, :);
X_t(5, :) = X(2, :).*X(2, :);
X_t(6, :) = X(3, :).*X(3, :);

[xt1_size, xt2_size] = size(X_test);
X_test = vertcat(ones(1, xt2_size), X_test);
X_test_t(1, :) = X_test(1, :);
X_test_t(2, :) = X_test(2, :);
X_test_t(3, :) = X_test(3, :);
X_test_t(4, :) = X_test(2, :).*X_test(3, :);
X_test_t(5, :) = X_test(2, :).*X_test(2, :);
X_test_t(6, :) = X_test(3, :).*X_test(3, :);
w = linear_regression(X_t(2:6, :), y);


pre_y = w' * X_t;
% train_error = (y - pre_y) * (y - pre_y)';
pre_y = sign(pre_y);
[aaa, train_error] = size(find(y~=pre_y));
E_train = train_error/x2_size;

pre_y_test = w' * X_test_t;
% test_error = (y_test - pre_y_test)* (y_test - pre_y_test)';
pre_y_test = sign(pre_y_test);
[aaa, test_error] = size(find(y_test~=pre_y_test));
E_test = test_error/xt2_size;
% Compute training, testing error
disp('------------Linear Regression with poly_fit of transform----------');
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);


%% Part6: Logistic Regression
nRep = 100; % number of replicates
nTrain = 100; % number of training data
w_g_sum = 0;
% J_sum  = 0;
E_train_sum = 0;
E_test_sum = 0;
for i = 1:nRep
    [Xlg, ylg, w_flg] = mkdata(large_num);
    X = Xlg(:, 1:nTrain);
    y = ylg(1, 1:nTrain);
    [w_g, J] = logistic(X, y);
    w_g_sum = w_g_sum + w_g;
    
    % compute training error
    au_x = vertcat(ones(1, nTrain), X);
    pre_y = 1.0./(1+exp((-w_g' * au_x)));
    % J_sum = J_sum + J;
    pre_y = sign(pre_y-0.5);
    [aaa, train_error] = size(find(y~=pre_y));
    E_train_sum = E_train_sum + train_error;
    
    % Compute testing error
    t_Xlg = Xlg(:, (nTrain + 1):large_num);
    t_ylg = ylg(1, (nTrain + 1):large_num);
    au_x_tg = vertcat(ones(1, (large_num-nTrain)), t_Xlg);
    pre_y_lg = 1.0./(1.0+exp((-w_g' * au_x_tg)));
    pre_y_lg = sign(pre_y_lg-0.5);
    [aaa, test_error] = size(find(t_ylg~=pre_y_lg));
    E_test_sum = E_test_sum + test_error;
    
end

E_train = E_train_sum / nRep/nTrain;
E_test = E_test_sum/nRep/(large_num-nTrain);
w_g_av = w_g_sum/nRep;
% !!!注意： 这里的logistic regression 暂时还有问题，需要再调整

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_flg, w_g, 'Logistic Regression');

%% Part7: Logistic Regression: noisy
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 10000; % number of training data
E_train_sum = 0;
E_test_sum = 0;
for i = 1:nRep
    [Xlgn, ylgn, w_flgn] = mkdata(large_num, 'noisy');
    X = Xlgn(:, 1:nTrain);
    y = ylgn(1, 1:nTrain);
    w_g = logistic(X, y);
    au_x = vertcat(ones(1, nTrain), X);
    % Compute training error
    pre_y = 1.0./(1+exp((-w_g' * au_x)));
    pre_y = sign(pre_y-0.5);
    [aaa, train_error] = size(find(y~=pre_y));
    E_train_sum = E_train_sum + train_error;
    
    % compute test error
    t_Xlgn = Xlgn(:, (nTrain + 1):large_num);
    t_ylgn = ylgn(1, (nTrain + 1):large_num);
    au_x_tgn = vertcat(ones(1, (large_num-nTrain)), t_Xlgn);
    pre_y_lgn = 1.0./(1+exp((-w_g' * au_x_tgn)));
    pre_y_lgn = sign(pre_y_lgn-0.5);
    [aaa, test_error] = size(find(t_ylgn~=pre_y_lgn));
    E_test_sum = E_test_sum + test_error;
end

E_train = E_train_sum / nRep/nTrain; 
E_test = E_test_sum/nRep/(large_num-nTrain);
w_g_av = w_g_sum/nRep;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_flgn, w_g, 'Logistic Regression: noisy');

%% Part8: SVM
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
sc_sum = 0;
E_train_sum = 0;
E_test_sum = 0;
for i = 1: nRep
    [Xs, ys, w_fs] = mkdata(large_num);
    X = Xs(:, 1:nTrain);
    y = ys(1, 1:nTrain);
    [w_g, num_sc] = svm(X, y);
    % Compute training error
    au_x = vertcat(ones(1, nTrain), X);
    pre_y = w_g' * au_x;
    pre_y = sign(pre_y);
    [aaa, train_error] = size(find(y~=pre_y));
    E_train_sum = E_train_sum + train_error;
    % compute test error
    t_Xs = Xs(:, (nTrain + 1):large_num);
    t_ys = ys(1, (nTrain + 1):large_num);
    au_x_ts = vertcat(ones(1, (large_num-nTrain)), t_Xs);
    pre_y_ts = w_g' * au_x_ts;
    pre_y_ts = sign(pre_y_ts);
    [aaa, wrong_num] = size(find(t_ys~=pre_y_ts));
    E_test_sum = E_test_sum + wrong_num;
    % Sum up number of support vectors
    sc_sum = sc_sum + num_sc;
end
E_train = E_train / nRep/nTrain; 
E_test = E_test_sum/nRep/(large_num-nTrain);
w_g_av = w_g_sum/nRep;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('The average number of support vector is %f.\n', sc_sum/nRep);
plotdata(X, y, w_fs, w_g, 'SVM');
