function [f,g] = CNN_Loss(w,X,y,kernel_size,nHidden,nLabels,lambda)
nInstances = size(X,1);
nVars = 144;
% Form Weights
offset = kernel_size * kernel_size;
convWeights=reshape(w(1:offset),kernel_size,kernel_size);
inputWeights = reshape(w(offset+1:offset + nVars * nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1)+offset;
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);


gConv = zeros(size(convWeights)); 
f = 0;
if nargout > 1
    gInput = zeros(size(inputWeights));
    gOutput = zeros(size(outputWeights));
end

% Compute Output
for i = 1:nInstances
    convInput = reshape(X(i,1:256),16,16);
    convOutput = conv2(convInput,convWeights,'valid');
    Z = reshape(convOutput,1,144);
    ip = Z * inputWeights;
    fp = tanh(ip);
    z = fp * outputWeights;
    yhat = exp(z) ./ sum(exp(z));
    
    relativeErr = -log(yhat(y(i,:) == 1));
    f = f + relativeErr;
    
    if nargout > 1
        err = yhat - (y(i,:) == 1);
        gOutput = gOutput + fp' * err + 2*lambda * outputWeights;
        backprop = err * (repmat(sech(ip),nLabels,1).^2.*outputWeights');
        gInput = gInput + Z'* backprop + 2*lambda*inputWeights;
        bias=[];
        bias= inputWeights;
        [,col]=size(bias);
        bias(:,col)=0;
        gInput=gInput+lambda*bias;
        backprop = backprop * inputWeights';
        reverseX = reshape(X(i,end:-1:1),16,16);
        backprop = reshape(backprop,12,12);
        gConv = gConv + conv2(reverseX, backprop, 'valid') + 2*lambda*convWeights;
    end

end

    
% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:kernel_size*kernel_size) = gConv(:);
    offset = kernel_size * kernel_size;
    g(offset+1:offset + nVars*nHidden(1)) = gInput(:);
    offset = offset + nVars*nHidden(1);
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end