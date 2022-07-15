addpath('../')
load digits.mat;

% the sample id, it start with size(X,1)+1 if we want to add more samples
flag = size(X,1)+1;

for i = 1:size(X,1)
        
        % reshape into 16*16 image
        img = reshape(X(i,:),16,16)/255;
        % transformation randomly.
        p = unifrnd (0,1); 
        if p<0.12
            img = transformation(img)
            % add into trainnig set
            X(flag,:) = floor(255*img(:));
            y(flag,:) = y(i,:);
            flag = flag + 1;
        end
end

% save the new data 
save(['transformed_digits.mat'],'Xtest','ytest','X','y','Xvalid','yvalid');

