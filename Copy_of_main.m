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
% �����ظ����л��������������delete(gcp('nocreate'))���Խ����
%�Ѿ���delete(gcp('nocreate'))�ӵ�����

delete(gcp('nocreate'));
close all;%�ر�����figure����
clear all;%��������ռ�����б�����������mex�ļ�
clc;%��������

%-------
% Paths.��ʵ�������ļ����ӵ�������
%-------
addpath('modelspecific');
addpath('mexfiles');
addpath('multigs');

%-------------------
% Compile Mex files.����Mex�ļ�
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
% Setup VLFeat toolbox.����VLFeat������
%----------------------
cd vlfeat-0.9.14/toolbox;
feval('vl_setup');
cd ../..;

%---------------------------------------------
% Check if we are already running in parallel.���м��㺯��
%---------------------------------------------
poolsize = parpool('local');
if poolsize == 0 %if not, we attempt to do it:
    parpool open;
end

%-------------------------
% User defined parameters.
%-------------------------
% Global model specific function handlers.ȫ��ģ�ͺ���
clear global;
global fitfn resfn degenfn psize numpar   % ������Щ����Ϊȫ�ֱ���
fitfn = 'homography_fit';
resfn = 'homography_res';
degenfn = 'homography_degen';
psize   = 4;
numpar  = 9;

M     = 500;  % Number of hypotheses for RANSAC.RANSAC�㷨���������������
thr   = 0.1;  % RANSAC threshold.

C1 = 100; % Resolution/grid-size for the mapping function in MDLT (C1 x C2).MDLT��ӳ�亯���ķֱ���/�����С
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

        % Load images and SIFT matches for temple data.����SIFTdata�ļ����е�����
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
            [ kp1,ds1 ] = vl_sift(single(rgb2gray(img1)),'PeakThresh', 0,'edgethresh',500);%����ͼ��1��������%kp1�ؼ������꣬ds1�ؼ���������
            [ kp2,ds2 ] = vl_sift(single(rgb2gray(img2)),'PeakThresh', 0,'edgethresh',500);%����ͼ��2��������
            matches   = vl_ubcmatch(ds1,ds2);%����ƥ����%vl_ubcmatchʹ�������ƥ��Ѱ�������Ƶ�������
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
%     % Load images and SIFT matches for temple data.����SIFTdata�ļ����е�����
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

% Normalise point distribution.%��ƥ�����й淶������
fprintf('  Normalising point distribution...');tic;
%���������ͼ����ƥ�������꣬ͬʱ������������ֵ꣨Ϊ1��
data_orig = [ kp1(1:2,matches(1,:)) ; ones(1,size(matches,2)) ; kp2(1:2,matches(2,:)) ; ones(1,size(matches,2)) ];
[ dat_norm_img1,T1 ] = normalise2dpts(data_orig(1:3,:));
[ dat_norm_img2,T2 ] = normalise2dpts(data_orig(4:6,:));
data_norm = [ dat_norm_img1 ; dat_norm_img2 ];%�ϲ��淶����ĵ�
fprintf('done (%fs)\n',toc);
% data_orig ������ǣ�
% 
% �� 1 �У���һ��ͼ��� x ����
% �� 2 �У���һ��ͼ��� y ����
% �� 3 �У�ȫΪ 1 ����
% �� 4 �У��ڶ���ͼ��� x ����
% �� 5 �У��ڶ���ͼ��� y ����
% �� 6 �У�ȫΪ 1 ����
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
% Multi-GS  ʵ���� RANSAC �㷨�Ľ�����ӻ���չʾ�˹ؼ�����ڵ��ƥ�����
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
TL = Hg\[1;1;1];%�õ��������ʽ
TL = round([ TL(1)/TL(3) ; TL(2)/TL(3) ]);%��λ�
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

% Canvas size.%���㻭����С
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

% Blending images by simple average (linear blending)ͼ���ں�
fprintf('  Homography linear image blending (averaging)...');tic;
linear_hom = imageblending(warped_img1,warped_img2);
fprintf('done (%fs)\n',toc);
figure;
imshow(linear_hom);
title('Image Stitching with global homography');
%t2 = clock;
%fprintf('global DLT done (%fs)\n',t2-t1);
%-------------------------
% Moving DLT (projective).
%-------------------------
fprintf('As-Projective-As-Possible Moving DLT on inliers\n');

% Image keypoints coordinates.
Kp = [data_orig(1,inliers)' data_orig(2,inliers)'];

% Generating mesh for MDLT.
fprintf('> Generating mesh for MDLT...');tic;
filename  =  'pointsInformation.xls';
[~, ~, raw] = xlsread(filename); % ��ȡ�������ݣ�raw ��ԭʼ����
data = cell2mat(raw(2:end, 1:4)); % �ӵڶ��п�ʼ��ѡ��ǰ���в�ת��Ϊ����
%disp(data);
%[ X,Y ] = meshgrid(linspace(1,cw,C1),linspace(1,ch,C2));%����һ������linspace(1, cw, C1) �� linspace(1, ch, C2) ������ X �� Y ������ꡣ
%fprintf('done (%fs)\n',toc);

% Mesh (cells) vertices' coordinates.
%Mv = [X(:)-off(1), Y(:)-off(2)];%����������ÿ��������꣬������ƫ�ơ�
Mv_new = [data(:,1)-off(1),data(:,2)-off(2),data(:,3),data(:,4)];
% Perform Moving DLT
fprintf('  Moving DLT main loop...');tic;
Hmdlt = zeros(size(Mv_new,1),9);%��ʼ��Hmdlt�������ڴ洢ÿ��������ת������
parfor i=1:size(Mv_new,1)%ʹ�� parfor ѭ��ʵ�ֲ��м��㣬����ÿ�������ı任��
    
    % Obtain kernel
    Gki = exp(-pdist2(Mv_new(i,1:2),Kp)./sigma^2);%pdist2 �������㵱ǰ���������֪�� Kp �ľ��룬���Ӧ��ָ����������Ȩ�� Gki��

    % Capping/offsetting kernel
    Wi = max(gamma,Gki); %ȷ��Ȩ�ز�С�� gamma��
    
    % This function receives W and A and obtains the least significant 
    % right singular vector of W*A by means of SVD on WA (Weighted SVD).
    v = wsvd(Wi,A);%���� wsvd �������м�Ȩ SVD���Ի�ȡ�任�������С����������������
    h = reshape(v,3,3)';        
    
    %ͨ�� D2\h*D1 �� T2\h*T1 ���о����ȥ��������ȥ��һ�����Իָ�ԭʼ���ݵĳ߶ȡ�
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
%[warped_img2] = imagewarping(double(ch),double(cw),double(img2),Hmdlt,double(off),X(1,:),Y(:,1)');
[warped_img2] = imagewarping(double(ch),double(cw),double(img2),Hmdlt,double(off),Mv_new(:,1)',Mv_new(:,2)',Mv_new(:,3)',Mv_new(:,4)');

warped_img2 = reshape(uint8(warped_img2),size(warped_img2,1),size(warped_img2,2)/3,3);
fprintf('done (%fs)\n',toc);

% Blending images by averaging (linear blending)
fprintf('  Moving DLT linear image blending (averaging)...');tic;
linear_mdlt = imageblending(warped_img1,warped_img2);
fprintf('done (%fs)\n',toc);
figure;
imshow(linear_mdlt);
title('As-Projective-As-Possible Image Stitching with Moving DLT������������NEW');
%t3 = clock;
%fprintf('all done (%fs)\n',t3-t1);
%t2 = clock;

fprintf('> Finished!.\n');