load X_test.mat
load X_train.mat
load y_test.mat
load y_train.mat

var = size(y_train);
label_p = cell(1,var(2));
%for prediction%
for i= 1: var(2)
    svm_polynomial = fitcsvm(X_train,y_train(:,i),'Standardize',true,'KernelFunction','polynomial','PolynomialOrder',3);
    label_p{i} = predict(svm_polynomial,X_test);
end

%Converting cell to matrix%
n = zeros(size(y_test,1),var(2));
for i=1 : var(2)
    n(:,i) = cell2mat(label_p(i));
end

%Calculating the jaccard similarity%
testSet = zeros(size(y_test,1));
for i =1: size(y_test,1)
     testSet(i) = pdist2(y_test(i,:),n(i,:),'jaccard');
end
count = 0;

%calculating accuracy%
for i= 1: size(y_test,1)
    count = count+ testSet(i);
end
accuracy_svm = count/size(y_test,1) * 100 ;
result_svm = ['The accuracy_svm =', num2str(accuracy_svm)];
disp(result_svm)