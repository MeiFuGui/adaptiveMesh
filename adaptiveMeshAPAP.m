% 代码重复运行会出错，输入命令delete(gcp('nocreate'))可以解决。
%已经将delete(gcp('nocreate'))加到首行

delete(gcp('nocreate'));
close all;%关闭所有figure窗口
clear all;%清除工作空间的所有变量、函数和mex文件
clc;%清除命令窗口
 
filename = 'F:\imagesData\LPC-01_library\result\10_50_0.4\pointsInformation.xls';  %自适应划分网格结果
path1 =    'F:\imagesData\LPC-01_library\01.jpg';                                  %参考图像
path2 =    'F:\imagesData\LPC-01_library\02.jpg';                                  %目标图像
path3 =    'F:\imagesData\LPC-01_library\result\global_result.jpg';                %全局单应性变换结果
path4 =    'F:\imagesData\LPC-01_library\result\10_50_0.4\adaptiveMeshResult.jpg'; %自适应网格APAP图像拼接结果

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



% [~, ~, raw] = xlsread(filename); % 读取所有数据，raw 是原始数据
% data = cell2mat(raw(2:end, 1:4)); % 从第二行开始，选择前四列并转换为矩阵

gamma = 0.01; % Normalizer for Moving DLT. (0.0015-0.1 are usually good numbers).
sigma = 8.5;  % Bandwidth for Moving DLT. (Between 8-12 are good numbers).   
scale = 1;    % Scale of input images (maybe for large images you would like to use a smaller scale).
% Images to stitch.
%------------------

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
    title('Source Images');
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
warped_img2 = imagewarping_new(double(ch),double(cw),double(img2),Hg,double(off));
warped_img2 = reshape(uint8(warped_img2),size(warped_img2,1),size(warped_img2,2)/3,3);
fprintf('done (%fs)\n',toc);

% Blending images by simple average (linear blending)图像融合
fprintf('  Homography linear image blending (averaging)...');tic;
linear_hom = imageblending(warped_img1,warped_img2);
%linear_hom(linear_hom == 0) = 255;%把黑色背景变白色
fprintf('done (%fs)\n',toc);
figure;
imshow(linear_hom);
title('Image Stitching with global homography');
imwrite(linear_hom , path3);


%t2 = clock;
%fprintf('global DLT done (%fs)\n',t2-t1);

% Moving DLT (projective).
%-------------------------
fprintf('As-Projective-As-Possible Moving DLT on inliers\n');

% Image keypoints coordinates.
Kp = [data_orig(1,inliers)' data_orig(2,inliers)'];

% Generating mesh for MDLT.
fprintf('> Generating mesh for MDLT...');tic;
%filename  =  'pointsInformation.xls';
[~, ~, raw] = xlsread(filename); % 读取所有数据，raw 是原始数据
data = cell2mat(raw(2:end, 1:4)); % 从第二行开始，选择前四列并转换为矩阵
%disp(data);
%[ X,Y ] = meshgrid(linspace(1,cw,C1),linspace(1,ch,C2));%创建一个网格，linspace(1, cw, C1) 和 linspace(1, ch, C2) 生成了 X 和 Y 轴的坐标。
%fprintf('done (%fs)\n',toc);

% Mesh (cells) vertices' coordinates.
%Mv = [X(:)-off(1), Y(:)-off(2)];%计算网格中每个点的坐标，并进行偏移。
Mv = [data(:,1)-off(1),data(:,2)-off(2),data(:,3),data(:,4)];
% Perform Moving DLT
fprintf('  Moving DLT main loop...');tic;
Hmdlt = zeros(size(Mv,1),9);%初始化Hmdlt矩阵，用于存储每个网格点的转换矩阵
parfor i=1:size(Mv,1)%使用 parfor 循环实现并行计算，处理每个网格点的变换。
    
    % Obtain kernel
    Gki = exp(-pdist2(Mv(i,1:2),Kp)./sigma^2);%pdist2 函数计算当前网格点与已知点 Kp 的距离，随后应用指数函数生成权重 Gki。

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
fprintf('Adaptive-Mesh-As-Projective-As-Possible Image stitching with Moving DLT and linear blending\n');
% Warping images with Moving DLT.
fprintf('> Warping images with Moving DLT...');tic;
warped_img1 = uint8(zeros(ch,cw,3));
warped_img1(off(2):(off(2)+size(img1,1)-1),off(1):(off(1)+size(img1,2)-1),:) = img1;
%[warped_img2] = imagewarping_new(double(ch),double(cw),double(img2),Hmdlt,double(off),X(1,:),Y(:,1)');
[warped_img2] = imagewarping_adaptive(double(ch),double(cw),double(img2),Hmdlt,double(off),data(:,1),data(:,2),data(:,3),data(:,4));

%[warped_img2] = imagewarping_new(double(ch),double(cw),double(img2),Hg,double(off),X(1,:),Y(:,1)');

%[warped_img2] = imagewarping(double(ch),double(cw),double(img2),Hmdlt,double(off),Mv_new(:,1)',Mv_new(:,2)',Mv_new(:,3)',Mv_new(:,4)');

warped_img2 = reshape(uint8(warped_img2),size(warped_img2,1),size(warped_img2,2)/3,3);
fprintf('done (%fs)\n',toc);
figure;
imshow(warped_img2);
title('Warp Image');
% Blending images by averaging (linear blending)
fprintf('  Moving DLT linear image blending (averaging)...');tic;
linear_mdlt = imageblending_Feathering(warped_img1,warped_img2);
linear_mdlt(linear_mdlt == 0) = 255;
fprintf('done (%fs)\n',toc);
figure;
imshow(linear_mdlt);
imwrite(linear_mdlt , path4);
title('Adaptive-Mesh-As-Projective-As-Possible Image Stitching with Moving DLT');

fprintf('> Finished!.\n');