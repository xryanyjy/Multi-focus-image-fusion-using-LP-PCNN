clc
clear
close all
% x1 = [1 2 -4; 1 1 1; -3 2 2; 0 0 0]
% y= norma_image(x1)
%-----------------------
I = imread('\clockA.gif');
I1 = im2double(I);%把图像数据类型转换为double类型并返回
I = imread('\clockB.gif');
I2 = im2double(I);
%--------------------------
I_blur_Left = I1(160:436,1:222);%表示位置坐标
I_clear_Left = I2(160:436,1:222);
I_blur_Right = I1(30:406,223:end);
I_clear_Right = I2(30:436,223:end);
%--------------------------显示图像区域------------------
figure
subplot(2,3,1);imshow(I1,[]);%figure3个图排成一行，一共两行，第1个位置
subplot(2,3,4);imshow(I2,[]);%imshow显示图像，[]默认表示[min(I(;)),max...]灰度值
subplot(2,3,2);imshow(I_blur_Left,[]);
subplot(2,3,5);imshow(I_clear_Left,[]);
subplot(2,3,3);imshow(I_blur_Right,[]);
subplot(2,3,6);imshow(I_clear_Right,[]);%figure1
%-------------------参数设置------------------------******************************
beta = 0.01;%链接强度
N0 = 200;%周期振荡稳定点
alpha_F = 4.61/N0 + 0.001;%F通道迭代衰减指数
V_T = 1/(1-exp(-alpha_F))+1;%阈值幅值
V_L = 1;
V_F = 0.1;
C = 5;%振荡区间
alpha_T =(log(1+V_T*(1-exp(-alpha_F))))/C;
alpha_L = 8.7/(ceil((log(exp(-alpha_T)+V_T*(1-exp(-alpha_F))))/alpha_T)+1)+0.001;
theata0 = 1.2;
%----------------------------------------------参数数组化****************************
P(1)= V_L;
P(2) = alpha_L;
P(3) = V_F;
P(4) = alpha_F;
P(5) = V_T;
P(6) = alpha_T;
P(7) = theata0;
P(8) = beta;
%----------------------------*****************************-------------------------------
N = 200;%迭代次数
Fire_series1 = PCNN_oscillation(I1,N,P);
Fire_series11 = Fire_series1./max(Fire_series1(:));%归一化，方便计算
Fire_series2 = PCNN_oscillation(I2,N,P);
Fire_series22 = Fire_series2./max(Fire_series2(:));
%---------------------------------**********************************
xn = 10;
K = ones(xn,xn);
FA = (conv2(Fire_series11,K,'same'))./(xn.^2);
FA = abs(Fire_series11 - FA);%求绝对值函数
FB = (conv2(Fire_series22,K,'same'))./(xn.^2);
FB = abs(Fire_series22 - FB);
%-------------------------------------------------******************
sel_logical = double(FA >FB);
figure
imwrite(sel_logical,'E.bmp')
imshow(sel_logical,[]);%figure2
image_fushion = I1.*sel_logical + I2.*double(sel_logical==0);
figure
imwrite(image_fushion,'F.bmp')
imshow(image_fushion);%figure3

%**********************************************************
%[m,n] = size(I);%返回图像I矩阵的行数和列数
%x = 1:m;
%y = 1:n;
%[X,Y] = meshgrid(x,y);%生成网格采样点函数
%figure
% surf(X,Y,FA);
%mesh(FA);%生成有X,Y,Z指定的网线面

A=imread('clockA.gif');
B=imread('clockB.gif');
F=imread('F.bmp');
FD=double(F);
averEntropy1=averEntropy(A)%计算信息熵,unit8格式
averEntropy2=averEntropy(B)
averEntropy3=averEntropy(F)

%NHOOD = [1,1,1];

%计算空间频率
SF=space_frequency(F)

aveGrad1 = AG(A)%计算平均梯度，double格式
aveGrad2 = AG(B)
aveGrad3 = avegrad(F);
avergradF=AG(F)


%计算互信息，unit8格式
MAF=mutinf(A,F);
MBF=mutinf(A,F);
MABF=MAF+MBF





