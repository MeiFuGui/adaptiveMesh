clc;
clear all;
close all;

% ָ���ļ�·�����ļ���
filename = 'pointsInformation.xls';

% ��ȡ Excel �ļ��е����ݣ��ӵڶ��п�ʼ
[~, ~, data] = xlsread(filename);

% �����ݴӵڶ��п�ʼ�����һ����ȡ����
data_matrix = data(2:end, :);

% �����ֻ��Ҫ��ֵ���ݣ����Һ����ı�������ʹ�����´��룺
num_data = cell2mat(data_matrix);

fprintf('done (%fs)\n',toc);