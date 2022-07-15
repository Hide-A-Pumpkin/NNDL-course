function [f,g] = SoftmaxLoss(w,X,y,nHidden,nLabels,lambda)
%feedforward neural network
[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)),nHidden(h-1)+1,nHidden(h));
  offset = offset+(nHidden(h-1)+1)*nHidden(h);
end
hiddenWeights{length(nHidden)} = w(offset+1:offset+(nHidden(end)+1)*nLabels);
hiddenWeights{length(nHidden)} = reshape(hiddenWeights{length(nHidden)},nHidden(end)+1,nLabels);
% saves in hiddenweights in the last

f = 0;
% Compute Output
for i = 1:nInstances
    ip{1} = X(i,:)*inputWeights;
    fp{1} = [1,tanh(ip{1})];
    f = f + lambda * norm(inputWeights,'fro');
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = [1,tanh(ip{h})];
        f = f + lambda * norm(hiddenWeights{h-1},'fro');
    end
    yi = fp{end}*hiddenWeights{end};
    yhat = exp(yi)./sum(exp(yi));
    f = f + lambda * norm(hiddenWeights{end},'fro');
    f = f + (- log(yhat(y(i))));

    if nargout > 1
%         err = 2*relativeErr;
        err = yhat;
        err(y(i)) = err(y(i)) - 1;
        for h = length(nHidden):-1:1
            gHidden{h} = fp{h}'* err;% equals gOutput 
            err = sech(ip{h}).^2 .* (err * hiddenWeights{h}(2:nHidden(h)+1,:)');
        end
        gInput = X(i,:)' * err+ 2 * lambda * inputWeights;
    end
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)) = gHidden{h-1};
        offset = offset+(nHidden(h-1)+1)*nHidden(h);
    end
    g(offset+1:offset+(nHidden(end)+1)*nLabels) = gHidden{end}(:);
end





