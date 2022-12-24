%using HOG features to detect real and fake fingerprint
close all
clear all
clc
%read input images(for train)
imagepath5='train';
filelist5=dir(fullfile(imagepath5,'*.bmp'));
list5={filelist5.name};
for i=1:length(list5)
    img5{i,1}=imresize(imread(fullfile(imagepath5,list5{i})),[96 96]); 
    
end
data_train=[img5];
xx=[];
for i=1:120
    input1=data_train{i,1};
    %extract HOG features
[hog1{i,1},visualization{i,1}] = extractHOGFeatures(input1,'CellSize',[12 12]);
xx=[xx,reshape(hog1{i,1},1764,1)];
end
xdata=double([xx]');
%label
for q=1:60
    group{q,1}='real';
end
for q=61:120
    group{q,1}='fake';
end
%svm struct
svmStruct= svmtrain(xdata,group,'kernel_function','rbf','rbf_sigma',35,'showplot',false);
%testing 
%read input images
%input images for testing
imagepath1='test';
filelist1=dir(fullfile(imagepath1,'*.bmp'));
list1={filelist1.name};
for i=1:length(list1)
    img1{i,1}=imresize(imread(fullfile(imagepath1,list1{i})),[96 96]); 
    
end
data_test=[img1];
for i=1:400
    noisy_image{i,1}=awgn(double(img1{i,1}),-28);
end
zz=[];
for i=1:400
    input11=noisy_image{i,1};
    %extract HOG features
[hog12{i,1},visualization{i,1}] = extractHOGFeatures(input11,'CellSize',[12 12]);
zz=[zz,reshape(hog12{i,1},1764,1)];
end
sample=double([zz]');
%svm test
Test = svmclassify(svmStruct,sample,'showplot',false);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%f-score
for i=1:300
    actual{i,1}=[0];
end
for i=301:400
    actual{i,1}=[1];
end
for i=1:400
    if Test{i,1}=='real'
        predicted{i,1}=[0];
    else
        predicted{i,1}=[1];
    end
end
ACTUAL=(cell2mat(actual));
PREDICTED=(cell2mat(predicted));
result=fscore(ACTUAL,PREDICTED);
