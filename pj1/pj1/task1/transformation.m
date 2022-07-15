function [new_img] = transformation(img)
[row,col] = size(img);
    
% calculate corresponding position of each pixel in new image
p = unifrnd (0,1); 
% transform
if p<=0.3
    trans = 4*rand(1,2)-2;   
    for i = 1:row
    for j = 1:col
        pos(i,j,:) = [i-trans(1),j-trans(2)];
    end
    end
end
% rotation
if p>0.3 && p<=0.6
    theta = 0.6*rand()-0.3;   
    for i = 1:row
        for j = 1:col
            x = j-col/2-0.5;
            y = i-row/2-0.5;
            pos(i,j,:) = [-x*sin(theta)+y*cos(theta)+row/2+0.5,x*cos(theta)+y*sin(theta)+row/2+0.5];
        end
    end
end
% resize
if p>0.6
    area = floor(14+5*rand(1,2));    
    center = [row,col]/2+0.5;
    for i = 1:row
        for j = 1:col
            pos(i,j,:) = [(i-center(1))*area(1)/row+center(1),(j-center(2))*area(2)/col+center(2)];
        end
    end
end
% get pixel value by linear interpolation
new_img = Bilinear_interpolation(img,pos);
end