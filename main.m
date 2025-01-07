%-------------------------------------------------------------------------
%       As-Projective-As-Possible Image Stitching with Moving DLT
%                           (Patent Pending)
%-------------------------------------------------------------------------
% The demo code in this package implements the non-rigid image stitching 
% method proposed in:
%
% "As-Projective-as-Possible Image Stitching with Moving DLT"
% Julio Zaragoza, Tat-Jun Chin, Michael Brown and David Suter,
% In Proc. Computer Vision and Pattern Recognition, Portland, Oregon, USA, 2013.
%
% Copyright (c) 2013-2014 Julio Zaragoza and Tat-Jun Chin
% School of Computer Science, The University of Adelaide, South Australia
% http://www.cs.adelaide.edu.au/~{jzaragoza,tjchin}
%
% The program is free for non-commercial academic use. Any commercial use
% is strictly prohibited without the authors' consent. Please acknowledge
% the authors by citing the above paper in any academic publications that
% have made use of this package or part of it.
%
% If you encounter any problems or questions please email to 
% jzaragoza@cs.adelaide.edu.au.
% 
% This program makes use of Peter Kovesi and Andrew Zisserman's MATLAB
% functions for multi-view geometry.
% (http://www.csse.uwa.edu.au/~pk/Research/MatlabFns/
%  http://www.robots.ox.ac.uk/~vgg/hzbook/code/).
%
% This program also makes use of Tat-Jun Chin's code for Accelerated  
% Hypothesis Generation for Multi-Structure Robust Fitting.
%
% We make use of eigen3's SVD decomposition (in our experience, it is 
% faster than MATLAB's SVD).
% 代码重复运行会出错，输入命令delete(gcp('nocreate'))可以解决。
%已经将delete(gcp('nocreate'))加到首行

delete(gcp('nocreate'));
close all;%关闭所有figure窗口
clear all;%清除工作空间的所有变量、函数和mex文件
clc;%清除命令窗口

%-------
% Paths.将实验所需文件添加到环境中
%-------
addpath('modelspecific');
addpath('mexfiles');
addpath('multigs');

%-------------------
% Compile Mex files.编译Mex文件
%-------------------
cd multigs;
if exist('computeIntersection','file')~=3
    mex computeIntersection.c; % <-- for multigs
end
cd ..;

cd mexfiles;
if exist('imagewarping','file')~=3
    mex ../imagewarping.cpp; 
end
if exist('wsvd','file')~=3
    mex ../wsvd.cpp; % We make use of eigen3's SVD in this file.
end
cd ..;

%----------------------
% Setup VLFeat toolbox.设置VLFeat工具箱
%----------------------
cd vlfeat-0.9.14/toolbox;
feval('vl_setup');
cd ../..;

%---------------------------------------------
% Check if we are already running in parallel.并行计算函数
%---------------------------------------------
poolsize = parpool('local');
if poolsize == 0 %if not, we attempt to do it:
    parpool open;
end

%-------------------------
% User defined parameters.
%-------------------------
% Global model specific function handlers.全局模型函数
clear global;
global fitfn resfn degenfn psize numpar   % 声明这些变量为全局变量
fitfn = 'homography_fit';
resfn = 'homography_res';
degenfn = 'homography_degen';
psize   = 4;
numpar  = 9;

M     = 500;  % Number of hypotheses for RANSAC.RANSAC算法中随机采样的数量
thr   = 0.1;  % RANSAC threshold.

C1 = 100; % Resolution/grid-size for the mapping function in MDLT (C1 x C2).MDLT中映射函数的分辨率/网格大小
C2 = 100;


switch input('Which images you want to stitch? [1 for ''temple''] [2 for ''railtracks''][3 for ''my_image''] ')
    case 1
        fprintf('> Stitching ''temple'' images\n');
        % In this implementation the weights are not calculated in the normalised 
        % space (but in the image space), therefore, these 2 following paramaters  
        % must be tuned in each case. 
        % If somebody wants to contribute to this code and calculate the weights in 
        % the normalised space so that the implementation is not too parameter-dependent, 
        % please, write me an email (jzaragoza@cs.adelaide.edu.au) and I'll be happy 
        % to talk with you :)
        gamma = 0.01; % Normalizer for Moving DLT. (0.0015-0.1 are usually good numbers).
        sigma = 8.5;  % Bandwidth for Moving DLT. (Between 8-12 are good numbers). 

        % Load images and SIFT matches for temple data.加载SIFTdata文件夹中的数据
        load 'SIFTdata/temple.mat'
    case 2
            fprintf('> Stitching ''railtracks'' images\n');    
            gamma = 0.0015; 
            sigma = 12; 

            % Load images and SIFT matches for railtracks data.
            load 'SIFTdata/railtracks.mat'
    case 3
            gamma = 0.1; % Normalizer for Moving DLT. (0.0015-0.1 are usually good numbers).
            sigma = 8.5;  % Bandwidth for Moving DLT. (Between 8-12 are good numbers).   
            scale = 1;    % Scale of input images (maybe for large images you would like to use a smaller scale).
            % Images to stitch.
            %------------------
            path1 = 'images/SEAGULL-01_books/01.png';
            path2 = 'images/SEAGULL-01_books/02.png';
            t1 = clock; 
            %-------------
            % Read images.
            %-------------
            fprintf('Read images and SIFT matching\n');tic;
            tic;
            fprintf('> Reading images...');tic;
            img1 = imresize(imread(sprintf('%s',path1)),scale);
            img2 = imresize(imread(sprintf('%s',path2)),scale);
            fprintf('done (%fs)\n',toc);
            
            %--------------------------------------
            % SIFT keypoint detection and matching.
            %--------------------------------------
            fprintf('  Keypoint detection and matching...');tic;
            [ kp1,ds1 ] = vl_sift(single(rgb2gray(img1)),'PeakThresh', 0,'edgethresh',500);%计算图像1的特征点%kp1关键点坐标，ds1关键点描述子
            [ kp2,ds2 ] = vl_sift(single(rgb2gray(img2)),'PeakThresh', 0,'edgethresh',500);%计算图像2的特征点
            matches   = vl_ubcmatch(ds1,ds2);%计算匹配点对%vl_ubcmatch使用最近邻匹配寻找最相似的描述子
            fprintf('done (%fs)\n',toc);
            t1 = clock;
end



% if input('Which images you want to stitch? [1 for ''temple''] [2 for ''railtracks''][3 for ''my_image''] ') == 1
%     fprintf('> Stitching ''temple'' images\n');
%     % In this implementation the weights are not calculated in the normalised 
%     % space (but in the image space), therefore, these 2 following paramaters  
%     % must be tuned in each case. 
%     % If somebody wants to contribute to this code and calculate the weights in 
%     % the normalised space so that the implementation is not too parameter-dependent, 
%     % please, write me an email (jzaragoza@cs.adelaide.edu.au) and I'll be happy 
%     % to talk with you :)
%     gamma = 0.01; % Normalizer for Moving DLT. (0.0015-0.1 are usually good numbers).
%     sigma = 8.5;  % Bandwidth for Moving DLT. (Between 8-12 are good numbers). 
%     
%     % Load images and SIFT matches for temple data.加载SIFTdata文件夹中的数据
%     load 'SIFTdata/temple.mat'
% else
%     fprintf('> Stitching ''railtracks'' images\n');    
%     gamma = 0.0015; 
%     sigma = 12; 
%     
%     % Load images and SIFT matches for railtracks data.
%     load 'SIFTdata/railtracks.mat'    
% end


%%%%%%%%%%%%%%%%%%%
% *** IMPORTANT ***
%%%%%%%%%%%%%%%%%%%
% If you want to try with your own images and make use of the VLFEAT
% library for SIFT keypoint detection and matching, **comment** the 
% previous IF/ELSE STATEMENT and **uncomment** the following code:
% 
% gamma = 0.1; % Normalizer for Moving DLT. (0.0015-0.1 are usually good numbers).
% sigma = 8.5;  % Bandwidth for Moving DLT. (Between 8-12 are good numbers).   
% scale = 1;    % Scale of input images (maybe for large images you would like to use a smaller scale).
% 
% %------------------
% % Images to stitch.
% %------------------
% path1 = 'images/case26/4.JPG';
% path2 = 'images/case26/5.JPG';
% 
% %-------------
% % Read images.
% %-------------
% fprintf('Read images and SIFT matching\n');tic;
% fprintf('> Reading images...');tic;
% img1 = imresize(imread(sprintf('%s',path1)),scale);
% img2 = imresize(imread(sprintf('%s',path2)),scale);
% fprintf('done (%fs)\n',toc);
% 
% %--------------------------------------
% % SIFT keypoint detection and matching.
% %--------------------------------------
% fprintf('  Keypoint detection and matching...');tic;
% [ kp1,ds1 ] = vl_sift(single(rgb2gray(img1)),'PeakThresh', 0,'edgethresh',500);
% [ kp2,ds2 ] = vl_sift(single(rgb2gray(img2)),'PeakThresh', 0,'edgethresh',500);
% matches   = vl_ubcmatch(ds1,ds2);
% fprintf('done (%fs)\n',toc);

% Normalise point distribution.%对匹配点进行规范化处理
fprintf('  Normalising point distribution...');tic;
%组合了两幅图像中匹配点的坐标，同时添加了齐次坐标（值为1）
data_orig = [ kp1(1:2,matches(1,:)) ; ones(1,size(matches,2)) ; kp2(1:2,matches(2,:)) ; ones(1,size(matches,2)) ];
[ dat_norm_img1,T1 ] = normalise2dpts(data_orig(1:3,:));
[ dat_norm_img2,T2 ] = normalise2dpts(data_orig(4:6,:));
data_norm = [ dat_norm_img1 ; dat_norm_img2 ];%合并规范化后的点
fprintf('done (%fs)\n',toc);
% data_orig 矩阵会是：
% 
% 第 1 行：第一幅图像的 x 坐标
% 第 2 行：第一幅图像的 y 坐标
% 第 3 行：全为 1 的行
% 第 4 行：第二幅图像的 x 坐标
% 第 5 行：第二幅图像的 y 坐标
% 第 6 行：全为 1 的行
if size(img1,1) == size(img2,1)    
    % Show input images.
    fprintf('  Showing input images...');tic;
    figure;
    imshow([img1,img2]);
    title('Input images');
    fprintf('done (%fs)\n',toc);
end

%-----------------
% Outlier removal.
%-----------------
fprintf('Outlier removal\n');tic;
% Multi-GS  实现了 RANSAC 算法的结果可视化，展示了关键点和内点的匹配情况
rng(0);
[ ~,res,~,~ ] = multigsSampling(100,data_norm,M,10);
con = sum(res<=thr);
[ ~, maxinx ] = max(con);
inliers = find(res(:,maxinx)<=thr);

if size(img1,1) == size(img2,1)
    % Show results of RANSAC.
    fprintf('  Showing results of RANSAC...');tic;
    figure;
    imshow([img1 img2]);
    hold on;
    plot(data_orig(1,:),data_orig(2,:),'ro','LineWidth',2);
    plot(data_orig(4,:)+size(img1,2),data_orig(5,:),'ro','LineWidth',2);
    for i=1:length(inliers)
        plot(data_orig(1,inliers(i)),data_orig(2,inliers(i)),'go','LineWidth',2);
        plot(data_orig(4,inliers(i))+size(img1,2),data_orig(5,inliers(i)),'go','LineWidth',2);
        plot([data_orig(1,inliers(i)) data_orig(4,inliers(i))+size(img1,2)],[data_orig(2,inliers(i)) data_orig(5,inliers(i))],'g-');
    end
    title('Ransac''s results');
    fprintf('done (%fs)\n',toc);
end

%-----------------------
% Global homography (H).
%-----------------------
fprintf('DLT (projective transform) on inliers\n');
% Refine homography using DLT on inliers.
fprintf('> Refining homography (H) using DLT...');tic;
[ h,A,D1,D2 ] = feval(fitfn,data_norm(:,inliers));
Hg = T2\(reshape(h,3,3)*T1);
fprintf('done (%fs)\n',toc);

%----------------------------------------------------
% Obtaining size of canvas (using global Homography).
%----------------------------------------------------
fprintf('Canvas size and offset (using global Homography)\n');
fprintf('> Getting canvas size...');tic;
% Map four corners of the right image.
TL = Hg\[1;1;1];%得到非齐次形式
TL = round([ TL(1)/TL(3) ; TL(2)/TL(3) ]);%齐次化
BL = Hg\[1;size(img2,1);1];
BL = round([ BL(1)/BL(3) ; BL(2)/BL(3) ]);
TR = Hg\[size(img2,2);1;1];
TR = round([ TR(1)/TR(3) ; TR(2)/TR(3) ]);
BR = Hg\[size(img2,2);size(img2,1);1];
BR = round([ BR(1)/BR(3) ; BR(2)/BR(3) ]);
fprintf('TL: (%f)',TL);
fprintf('BL: (%f)',BL);
fprintf('TR: (%f)',TR);
fprintf('BR: (%f)',BR);

% Canvas size.%计算画布大小
cw = max([1 size(img1,2) TL(1) BL(1) TR(1) BR(1)]) - min([1 size(img1,2) TL(1) BL(1) TR(1) BR(1)]) + 1;
ch = max([1 size(img1,1) TL(2) BL(2) TR(2) BR(2)]) - min([1 size(img1,1) TL(2) BL(2) TR(2) BR(2)]) + 1;
fprintf('done (%fs)\n',toc);

% Offset for left image.
fprintf('> Getting offset...');tic;
off = [ 1 - min([1 size(img1,2) TL(1) BL(1) TR(1) BR(1)]) + 1 ; 1 - min([1 size(img1,1) TL(2) BL(2) TR(2) BR(2)]) + 1 ];
fprintf('done (%fs)\n',toc);

%--------------------------------------------
% Image stitching with global homography (H).
%--------------------------------------------
% Warping source image with global homography 
fprintf('Image stitching with global homography (H) and linear blending\n');
fprintf('> Warping images by global homography...');tic;
warped_img1 = uint8(zeros(ch,cw,3));
warped_img1(off(2):(off(2)+size(img1,1)-1),off(1):(off(1)+size(img1,2)-1),:) = img1;
warped_img2 = imagewarping(double(ch),double(cw),double(img2),Hg,double(off));
warped_img2 = reshape(uint8(warped_img2),size(warped_img2,1),size(warped_img2,2)/3,3);
fprintf('done (%fs)\n',toc);

% Blending images by simple average (linear blending)图像融合
fprintf('  Homography linear image blending (averaging)...');tic;
linear_hom = imageblending(warped_img1,warped_img2);
fprintf('done (%fs)\n',toc);
figure;
imshow(linear_hom);
title('Image Stitching with global homography');
t2 = clock;
%fprintf('global DLT done (%fs)\n',t2-t1);
%-------------------------
% Moving DLT (projective).
%-------------------------
fprintf('As-Projective-As-Possible Moving DLT on inliers\n');

% Image keypoints coordinates.
Kp = [data_orig(1,inliers)' data_orig(2,inliers)'];

% 源网格拼接
% Generating mesh for MDLT.
fprintf('> Generating mesh for MDLT...');tic;
[ X,Y ] = meshgrid(linspace(1,cw,C1),linspace(1,ch,C2));%创建一个网格，linspace(1, cw, C1) 和 linspace(1, ch, C2) 生成了 X 和 Y 轴的坐标。




fprintf('done (%fs)\n',toc);

% Mesh (cells) vertices' coordinates.
Mv = [X(:)-off(1), Y(:)-off(2)];%计算网格中每个点的坐标，并进行偏移。

% Perform Moving DLT
fprintf('  Moving DLT main loop...');tic;
Hmdlt = zeros(size(Mv,1),9);%初始化Hmdlt矩阵，用于存储每个网格点的转换矩阵

parfor i=1:size(Mv,1)%使用 parfor 循环实现并行计算，处理每个网格点的变换。
    
    % Obtain kernel
    Gki = exp(-pdist2(Mv(i,:),Kp)./sigma^2);%pdist2 函数计算当前网格点与已知点 Kp 的距离，随后应用指数函数生成权重 Gki。
    % Gki = exp(-pdist2(num_data(i,1:2),Kp)./sigma^2);

    % Capping/offsetting kernel
    Wi = max(gamma,Gki); %确保权重不小于 gamma。
    
    % This function receives W and A and obtains the least significant 
    % right singular vector of W*A by means of SVD on WA (Weighted SVD).
    v = wsvd(Wi,A);%调用 wsvd 函数进行加权 SVD，以获取变换矩阵的最小显著右奇异向量。
    h = reshape(v,3,3)';        
    
    %通过 D2\h*D1 和 T2\h*T1 进行矩阵的去条件化和去归一化，以恢复原始数据的尺度。
    % De-condition
    h = D2\h*D1;

    % De-normalize
    h = T2\h*T1;
    
    Hmdlt(i,:) = h(:);
end
fprintf('done (%fs)\n',toc);

%---------------------------------
% Image stitching with Moving DLT.
%---------------------------------
fprintf('As-Projective-As-Possible Image stitching with Moving DLT and linear blending\n');
% Warping images with Moving DLT.
fprintf('> Warping images with Moving DLT...');tic;
warped_img1 = uint8(zeros(ch,cw,3));
warped_img1(off(2):(off(2)+size(img1,1)-1),off(1):(off(1)+size(img1,2)-1),:) = img1;
[warped_img2] = imagewarping_new(double(ch),double(cw),double(img2),Hmdlt,double(off),X(1,:),Y(:,1)');
warped_img2 = reshape(uint8(warped_img2),size(warped_img2,1),size(warped_img2,2)/3,3);
fprintf('done (%fs)\n',toc);
figure;
imshow(warped_img2);
% % 修改后的网格拼接
% % Generating mesh for MDLT.
% fprintf('> Generating mesh for MDLT...');tic;
% % [ X,Y ] = meshgrid(linspace(1,cw,C1),linspace(1,ch,C2));%创建一个网格，linspace(1, cw, C1) 和 linspace(1, ch, C2) 生成了 X 和 Y 轴的坐标。
% % 指定文件路径和文件名
% filename = 'pointsInformation.xls';
% 
% % 读取 Excel 文件中的数据，从第二行开始
% [~, ~, data] = xlsread(filename);
% 
% % 将数据从第二行开始到最后一行提取出来
% data_matrix = data(2:end, :);
% 
% % 如果你只需要数值数据，并且忽略文本，可以使用以下代码：
% num_data = cell2mat(data_matrix);
% % num_data = data_matrix;
% fprintf('done (%fs)\n',toc);
% 
% % Mesh (cells) vertices' coordinates.
% % Mv = [X(:)-off(1), Y(:)-off(2)];%计算网格中每个点的坐标，并进行偏移。
% num_data(:,1) = num_data(:,1)-off(1);
% num_data(:,2) = num_data(:,2)-off(2);
% % Perform Moving DLT
% fprintf('  Moving DLT main loop...');tic;
% % Hmdlt = zeros(size(Mv,1),9);%初始化Hmdlt矩阵，用于存储每个网格点的转换矩阵
% Hmdlt = zeros(size(num_data,1),9);
% parfor i=1:size(num_data,1)%使用 parfor 循环实现并行计算，处理每个网格点的变换。
%     
%     % Obtain kernel
%     % Gki = exp(-pdist2(Mv(i,:),Kp)./sigma^2);%pdist2 函数计算当前网格点与已知点 Kp 的距离，随后应用指数函数生成权重 Gki。
%     Gki = exp(-pdist2(num_data(i,1:2),Kp)./sigma^2);
% 
%     % Capping/offsetting kernel
%     Wi = max(gamma,Gki); %确保权重不小于 gamma。
%     
%     % This function receives W and A and obtains the least significant 
%     % right singular vector of W*A by means of SVD on WA (Weighted SVD).
%     v = wsvd(Wi,A);%调用 wsvd 函数进行加权 SVD，以获取变换矩阵的最小显著右奇异向量。
%     h = reshape(v,3,3)';        
%     
%     %通过 D2\h*D1 和 T2\h*T1 进行矩阵的去条件化和去归一化，以恢复原始数据的尺度。
%     % De-condition
%     h = D2\h*D1;
% 
%     % De-normalize
%     h = T2\h*T1;
%     
%     Hmdlt(i,:) = h(:);
% end
% fprintf('done (%fs)\n',toc);
% 
% %---------------------------------
% % Image stitching with Moving DLT.
% %---------------------------------
% fprintf('As-Projective-As-Possible Image stitching with Moving DLT and linear blending\n');
% % Warping images with Moving DLT.
% fprintf('> Warping images with Moving DLT...');tic;
% warped_img1 = uint8(zeros(ch,cw,3));
% warped_img1(off(2):(off(2)+size(img1,1)-1),off(1):(off(1)+size(img1,2)-1),:) = img1;
% % [warped_img2] = imagewarping(double(ch),double(cw),double(img2),Hmdlt,double(off),X(1,:),Y(:,1)');
% [warped_img2] = imagewarping_new(double(ch),double(cw),double(img2),Hmdlt,double(off),num_data(:,1)',num_data(:,2)',num_data(:,3)',num_data(:,4)');
% warped_img2 = reshape(uint8(warped_img2),size(warped_img2,1),size(warped_img2,2)/3,3);
% figure;
% imshow(warped_img2);
% fprintf('done (%fs)\n',toc);

% Blending images by averaging (linear blending)
fprintf('  Moving DLT linear image blending (averaging)...');tic;
linear_mdlt = imageblending(warped_img1,warped_img2);
fprintf('done (%fs)\n',toc);
figure;
imshow(linear_mdlt);
title('As-Projective-As-Possible Image Stitching with Moving DLT');
%t3 = clock;
%fprintf('all done (%fs)\n',t3-t1);
%t2 = clock;

fprintf('> Finished!.\n');
