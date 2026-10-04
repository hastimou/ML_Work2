from preprocessing import read_arff,preprocess
from models import svm_linear_algorithm,svm_rbf_algorithm,RF_algorithm
from calculate_accuracy import calculate_accuracy_time,show_results


for fold in range(10):
    print("\n" + "=" * 60)
    print("FOLD", fold)
    print("=" * 60)
    X_train_encoded,y_train,X_test_encoded,y_test=preprocess(fold)
    svm_linear_accuracies, svm_linear_times=svm_linear_algorithm(X_train_encoded,X_test_encoded,y_train,y_test,kernel="linear",C=1.0)
    svm_rbf_accuracies, svm_rbf_times=svm_rbf_algorithm(X_train_encoded,X_test_encoded,y_train,y_test,kernel="rbf",C=1.0,gamma="scale")
    RF_accuracies=RF_algorithm(X_train_encoded,X_test_encoded,y_train,y_test,n_estimators=300,criterion="gini",max_depth=None,max_features="sqrt",random_state=42,n_jobs=-1)
    mean_RF_accuracy, std_RF_accuracy,mean_svm_linear_accuracy,std_svm_linear_accuracy,mean_svm_rbf_accuracy,std_svm_rbf_accuracy,mean_svm_linear_time,mean_svm_rbf_time=calculate_accuracy_time(RF_accuracies,svm_linear_accuracies,svm_rbf_accuracies,svm_linear_times,svm_rbf_times)
    show_results(mean_RF_accuracy,std_RF_accuracy,mean_svm_linear_accuracy,std_svm_linear_accuracy,mean_svm_linear_time,mean_svm_rbf_accuracy,std_svm_rbf_accuracy,mean_svm_rbf_time)


