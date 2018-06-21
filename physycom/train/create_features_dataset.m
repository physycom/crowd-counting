close all
clear all
clc

%% set images path and ground-truth path
winSize = 100;
winStep = winSize - 1;
xStep = 50;
yStep = 50;

%% init pre-trained resnet-152 model
addpath(strcat(getenv('WORKSPACE'), '/matconvnet/matlab')); % matconvnet mex components
addpath(strcat(getenv('WORKSPACE'), '/crowd-counting'));    % resnet-152 dagnn-compliant weights
net = dagnn.DagNN.loadobj(load('imagenet-resnet-152-dag.mat'));
net.mode = 'test';
net.conserveMemory = 0;

%%
ImagePath = 'E:/Alessandro/Peoplebox/new/peoplebox_crowd_dataset/dataset_tiny/';
CountPath = 'E:/Alessandro/Peoplebox/new/peoplebox_crowd_dataset/dataset_tiny/';
image_regex = fullfile(ImagePath, '*.tiny.jpg');
count_regex = fullfile(CountPath, '*.tiny.csv');
image_in_dir = dir(image_regex);
count_in_dir = dir(count_regex);

n = length(image_in_dir);
features = cell(1, n);
counts = cell(1, n);

for i = 1 : n
    disp(image_in_dir(i).name)
    image_name = strsplit(image_in_dir(i).name, '.');
    check = regexp(count_in_dir(i).name, image_name(1), 'match');
    assert(~isempty(check));

    im = imread([image_in_dir(i).folder '/' image_in_dir(i).name]);
    [height, width, channel] = size(im);

    newHeight = round(height/50) * 50;
    newWidth = round(width/50) * 50;

    csvgt = csvread([count_in_dir(i).folder '\\' count_in_dir(i).name],1,0);
    location = csvgt(:,2:end);
    location(:, 1) = location(:, 1) / width * newWidth;
    location(:, 2) = location(:, 2) / height * newHeight;

    im = imresize(im, [newHeight, newWidth]);
    if channel == 1
        tmp = zeros(newHeight, newWidth, 3);
        tmp(:, :, 1) = im;
        tmp(:, :, 2) = im;
        tmp(:, :, 3) = im;
        im = tmp;
    end

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

            index =  (location(:, 1) > x - 0.5) & (location(:, 1) < x + winStep + 0.5 ) & (location(:, 2) > y - 0.5) & (location(:, 2) < y + winStep + 0.5);
            patchCount(row, column) = sum(index);

            x = x + xStep;
            column = column + 1;
        end
        y = y + yStep;
        row = row + 1;
    end

    features{i} = patchFeature;
    counts{i} = patchCount;
end

save dataset_tiny.mat features counts

