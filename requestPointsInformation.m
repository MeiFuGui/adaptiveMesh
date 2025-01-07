clc;
clear all;
close all;

% 指定文件路径和文件名
filename = 'pointsInformation.xls';

% 读取 Excel 文件中的数据，从第二行开始
[~, ~, data] = xlsread(filename);

% 将数据从第二行开始到最后一行提取出来
data_matrix = data(2:end, :);

% 如果你只需要数值数据，并且忽略文本，可以使用以下代码：
num_data = cell2mat(data_matrix);

fprintf('done (%fs)\n',toc);