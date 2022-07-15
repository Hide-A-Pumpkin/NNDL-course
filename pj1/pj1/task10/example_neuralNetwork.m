load transformed_digits.mat
tic
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);


% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
% nHidden = [10];
nHidden = [100];

% kernel size
kernel_size = 5;


% Count number of parameters and initialize weights 'w'
nParams = kernel_size * kernel_size + 144 * nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+(nHidden(h-1)) * nHidden(h);
end
nParams = nParams+(nHidden(end)) * nLabels;
w = randn(nParams,1);

% Train with stochastic gradient
maxIter = 200000;
stepSize = 1e-2;
lambda = 1e-3;
funObj = @(w,i)CNN_Loss(w,X(max(1,i-3):i,:), yExpanded(max(1,i-3):i,:),kernel_size,nHidden,nLabels,lambda);

iteration = [];
error = [];
key = 1;


for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = CNN_Predict(w,Xvalid,kernel_size, nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
        
        if key > sum(yhat~=yvalid)/t
            key = sum(yhat~=yvalid)/t;
            w_optimal = w;
        end
        
        iteration = [iteration, iter-1];
        error = [error,sum(yhat~=yvalid)/t];
    end
    
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w = w - stepSize*g;
end

% Evaluate test error
yhat = CNN_Predict(w,Xtest,kernel_size,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);


%plot error
figure(1);
plot(iteration,error,'-b');
title('error');
ylabel('validation error');
xlabel('iteration times');
axis([0,200000,0,1]);
toc