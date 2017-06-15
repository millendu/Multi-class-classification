%%%%% loading test data below %%%%%
load X_test.mat
load X_train.mat
load y_test.mat
load y_train.mat

%%%% transposing x_train and y_train data that is read %%%%
x_n = transpose(X_train);
y_n = transpose(y_train);

%%%%%% code to generate and train a model based on knn classifier %%%%%%
mdl = fitcknn(X_train,y_n,'NumNeighbors',7);
predict_knn = predict(mdl,X_test);
count_knn=0;

for i=1: numel(predict_knn)
    if predict_knn(i,1) == y_test(i,1)
        count_knn = count_knn + 1;
    end
end

%%% calculating accuracy of the model generated above %%%%
accuracy_knn = count_knn*100/numel(predict_knn);

%%%% code to generate and train a model based on svm classifier %%%%
Mdl = fitcecoc(X_train,y_n.','Learners',templateSVM('KernelFunction','polynomial','PolynomialOrder',2));

predict_svm = predict(Mdl,X_test(:,:));
y_test_trans = transpose(y_test);
output=y_test_trans(:,:).';
count_svm = 0;
for i = 1:numel(predict_svm)
    if (output(i,1) == predict_svm(i,1))
        count_svm = count_svm + 1;
    end
end

%%% calculating accuracy of the model generated above %%%%
accuracy_svm = (count_svm/numel(output)) * 100;

%%%% code to train a feedforward nueral network with 25 nuerons %%%%%
ffn = feedforwardnet(25);
y_vec = full(ind2vec(y_n));
ffn = train(ffn,x_n,y_vec);    %%%% training the feed forward  network with training dataset %%%%
view(ffn)   
x_trans1 = transpose(X_test);
y_new = ffn(x_trans1);             %%%% validating the generated neural network with test data %%%%
final_result = vec2ind(y_new);
vec_y_test_calc_trans = transpose(final_result);
count = 0;
for i = 1:numel(final_result)
    if final_result(i) == y_test(i)
        count = count +1;
    end
end

%%% calculating accuracy of the model generated above %%%%
accuracy_fnn = count/numel(final_result) *100;


%calculating Ensemble accuracy%
count = 0;
for i = 1:numel(y_test)
   A = [predict_svm(i,1),vec_y_test_calc_trans(i,1),predict_knn(i,1)];
   if (y_test(i,1) == mode(A))
        count = count + 1;
   end
   
end
accuracy_ensemble = (count/numel(y_test)) * 100 


%%% displaying the calculated accuracies %%%
result_knn = ['The accuracy_knn =', num2str(accuracy_knn)];
result_ffn = ['The accuracy_ffn =', num2str(accuracy_fnn)];
result_svm = ['The accuracy_svm =', num2str(accuracy_svm)];
result_ensemble = ['The accuracy_ensemble =', num2str(accuracy_ensemble)];



disp(result_knn);
disp(result_ffn);
disp(result_svm);
disp(result_ensemble);

      
    
