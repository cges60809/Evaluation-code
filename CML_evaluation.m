clc;clear all;close all;
%***********************************************%
% This code runs on the Market-1501 dataset.    %
% Please modify the path to your own folder.    %
% We use the mAP and hit-1 rate as evaluation   %
%***********************************************%
% if you find this code useful in your research, please kindly cite our
% paper as,
% Liang Zheng, Liyue Sheng, Lu Tian, Shengjin Wang, Jingdong Wang, and Qi Tian,
% Scalable Person Re-identification: A Benchmark, ICCV, 2015.

% Please download Market-1501 dataset and unzip it in the "dataset" folder.

%% load data for feature extraction
codebooksize = 350;
codebook = importdata(['data/codebook_' num2str(codebooksize) '.mat']);
par = importdata(['data/params_' num2str(codebooksize) '.mat']);
w2c = importdata('data/w2c.mat'); % used in CN extraction

%% add necessary paths
query_dir = '../dataset/Market-1501-v15.09.15/query/';% query directory
test_dir = '../dataset/Market-1501-v15.09.15/bounding_box_test/';% database directory
gt_dir = '../dataset/Market-1501-v15.09.15/gt_bbox/'; % directory of hand-drawn bounding boxes
addpath 'CM_curve/' % draw confusion matrix

%% import query features
Hist_query = importdata('feat_query.mat');
nQuery = size(Hist_query, 1);

%% import database features
Hist_test = importdata('feat_test.mat');
nTest = size(Hist_test, 1);

%% calculate the ID and camera for database images
test_files = dir([test_dir '*.jpg']);
testID = zeros(length(test_files), 1);
testCAM = zeros(length(test_files), 1);
if ~exist('data/testID.mat')
    for n = 1:length(test_files)
        img_name = test_files(n).name;
        if strcmp(img_name(1), '-') % junk images
            testID(n) = -1;
            testCAM(n) = str2num(img_name(5));
        else
            testID(n) = str2num(img_name(1:4));
            testCAM(n) = str2num(img_name(7));
        end
    end
    save('data/testID.mat', 'testID');
    save('data/testCAM.mat', 'testCAM');
else
    testID = importdata('data/testID.mat');
    testCAM = importdata('data/testCam.mat');    
end

%% calculate the ID and camera for query images
query_files = dir([query_dir '*.jpg']);
queryID = zeros(length(query_files), 1);
queryCAM = zeros(length(query_files), 1);
if ~exist('data/queryID.mat')
    for n = 1:length(query_files)
        img_name = query_files(n).name;
        if strcmp(img_name(1), '-') % junk images
            queryID(n) = -1;
            queryCAM(n) = str2num(img_name(5));
        else
            queryID(n) = str2num(img_name(1:4));
            queryCAM(n) = str2num(img_name(7));
        end
    end
    save('data/queryID.mat', 'queryID');
    save('data/queryCAM.mat', 'queryCAM');
else
    queryID = importdata('data/queryID.mat');
    queryCAM = importdata('data/queryCam.mat');    
end


%% search the database and calcuate re-id accuracy
ap = zeros(nQuery, 1); % average precision

CMC = zeros(nQuery, nTest);

r1 = 0; % rank 1 precision with single query

dist = sqdist(double(Hist_test.'), double(Hist_query.')); % distance calculate with single query. Note that Euclidean distance is equivalent to cosine distance if vectors are l2-normalized
%dist = (2-dist_sqt)./2; % cosine distance

knn = 1; % number of expanded queries. knn = 1 yields best result
queryCam = importdata('data/queryCam.mat'); % camera ID for each query
testCam = importdata('data/testCam.mat'); % camera ID for each database image

for k = 1:nQuery
    k
    % load groud truth for each query (good and junk)
    good_index = intersect(find(testID == queryID(k)), find(testCAM ~= queryCAM(k)))';% images with the same ID but different camera from the query
    junk_index1 = find(testID == -1);% images neither good nor bad in terms of bbox quality
    junk_index2 = intersect(find(testID == queryID(k)), find(testCAM == queryCAM(k))); % images with the same ID and the same camera as the query
    junk_index = [junk_index1; junk_index2]';
    tic
    score = dist(:, k);
    % sort database images according Euclidean distance
    [~, index] = sort(score, 'ascend');  % single query
    
    [ap(k), CMC(k, :)] = compute_AP(good_index, junk_index, index);% compute AP for single query
end

CMC = mean(CMC);
%% print result
fprintf('single query:     mAP = %f, r1 precision = %f\r\n', mean(ap), CMC(1));

%% plot CMC curves
figure;
s = 50;
CMC_curve = [ CMC ];
plot(1:s, CMC_curve(:, 1:s));




