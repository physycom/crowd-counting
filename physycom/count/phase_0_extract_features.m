%% init pre-trained resnet-152 model
addpath(strcat(getenv('WORKSPACE'), '/matconvnet/matlab')); % matconvnet mex components
addpath(strcat(getenv('WORKSPACE'), '/crowd-counting'));    % resnet-152 dagnn-compliant weights
net = dagnn.DagNN.loadobj(load('imagenet-resnet-152-dag.mat'));
net.mode = 'test';
net.conserveMemory = 0;

%% set patch dimension
stepw = 50;
steph = 50;
winw  = 100;
winh  = 100;

%% load image and resize
im = imread(imgpath);
[h, w, c] = size(im);
newh = round(h/steph) * steph;
neww = round(w/stepw) * stepw;
im = imresize(im, [newh, neww]);
if c == 1
  tmp = zeros(newh, neww, 3);
  tmp(:, :, 1) = im;
  tmp(:, :, 2) = im;
  tmp(:, :, 3) = im;
  im = tmp;
end

%% extract patches' features
y = 1;
row = 1;
features = zeros(newh / 50 - 1, neww / 50 - 1, 1000);
while(y + winh  <= newh)
  x = 1;
  column = 1;
  while(x + winw <= neww)
    img = im(y:y + winh, x:x + winw, :);% get image patch
    img = single(img);

    im_ = imresize(img, net.meta.normalization.imageSize(1:2));
    im_ = im_ - net.meta.normalization.averageImage ;

    net.eval({'data', im_});
    features(row ,column ,:) = reshape(net.vars(net.getVarIndex('fc1000')).value, 1, 1000);

    x = x + stepw;
    column = column + 1;
  end
  y = y + steph;
  row = row + 1;
end

%% dump results
tok=strsplit(imgpath, '.');
output = strcat(tok{1},'.phase_0.mat');
save(output,'features')
