close all
clear all
clc

addpath('E:\Alessandro\Codice\matconvnet\matlab')

%% set images path and ground-truth path
imagesPath = './';
groundPath = './';
n = 1; % images number
features = cell(1, n);
counts = cell(1, n);

winSize = 100;
winStep = winSize - 1;
xStep = 50;
yStep = 50;


%% init pre-trained resnet-152 model 
net = dagnn.DagNN.loadobj(load('../imagenet-resnet-152-dag.mat'));
net.mode = 'test';
net.conserveMemory = 0;

%% 
for i = 1 : n
    disp(i)
    im = imread([imagesPath 'conta_cappelli.jpg']);
    [height, width, channel] = size(im);
    
    newHeight = round(height/50) * 50;
    newWidth = round(width/50) * 50;
%    figure(1);
%    imshow(im)
    im = imresize(im, [newHeight, newWidth]);
%    figure(2);
%    imshow(im)
    if channel == 1
        im=cat(3,im(:, :, 1),im(:, :, 2),im(:, :, 3));
    end
%    figure(3);
%    imshow(im)
    
    y = 1;
    row = 1;
    patchFeature = zeros(newHeight / 50 - 1, newWidth/50 - 1, 1000);
    patchCount = zeros(newHeight / 50 - 1, newWidth/50 - 1);
    while(y + winStep  <= newHeight)
        x = 1;
        column = 1;
        while(x + winStep <= newWidth)
            img = im(y:y + winStep, x: x + winStep, :);% get image patch
            img = single(img);
            
            im_ = imresize(img, net.meta.normalization.imageSize(1:2));
            im_ = im_ - net.meta.normalization.averageImage ;
            
            net.eval({'data', im_});
            patchFeature(row ,column ,:) = reshape(net.vars(net.getVarIndex('fc1000')).value, 1, 1000);
            
            x = x + xStep;
            column = column + 1;
        end
        y = y + yStep;
        row = row + 1;
    end
    
    features{i} = patchFeature;
    counts{i} = patchCount;
end

save cambiaminomequandocapiscicosasono.mat features counts

