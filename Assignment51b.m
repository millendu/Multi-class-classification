
load X_test.mat
load X_train.mat
load y_test.mat
load y_train.mat

label = zeros(907,6);

%for prediction%
for i=1: 1: size (y_train,2)
    mdl = fitcsvm(X_train,y_train(:,i),'Standardize',true,'KernelFunction','gaussian','KernelScale','auto');
    label(:,i) = predict(mdl,X_test);
end

%calculating the count for accuracy%
count_svm = 0;

for i=1: 1: 907
    if label(i,:) == y_test(i,:)
        count_svm = count_svm+1;
    end
end

accuracy_svm = count_svm*100/(907)

result_svm = ['The accuracy_svm =', num2str(accuracy_svm)];
disp(result_svm)