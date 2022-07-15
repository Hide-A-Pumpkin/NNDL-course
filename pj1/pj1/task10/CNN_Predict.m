function [y] = CNN_Predict(w,X,kernel_size,nHidden,nLabels)
nInstances = size(X,1);
nVars = 144;
% Form Weights
convWeights = reshape(w(1:kernel_size * kernel_size),kernel_size,kernel_size);
offset = kernel_size * kernel_size;
inputWeights = reshape(w(offset+1:offset + nVars * nHidden(1)),nVars,nHidden(1));
offset = offset + nVars * nHidden(1);
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

% Compute Output
for i = 1:nInstances
    convInput = reshape(X(i,2:257),16,16);
    convOutput = conv2(convInput,convWeights,'valid');
    Z = reshape(convOutput,1,144);
    ip = Z * inputWeights;
    fp = tanh(ip);
    yi = fp * outputWeights;
    y(i,:) =exp(yi) ./ sum(exp(yi));
end
[v,y] = max(y,[],2);
