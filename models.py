from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import time



def svm_linear_algorithm(X_train_encoded,X_test_encoded,y_train,y_test,kernel="linear",C=1.0):
    svm_linear_accuracies = []
    svm_linear_times = []

    svm_linear = SVC(kernel=kernel,C=C)

    start_time = time.time()

    svm_linear.fit(X_train_encoded, y_train)

    y_pred_svm_linear = svm_linear.predict(X_test_encoded)

    end_time = time.time()

    svm_linear_time = end_time - start_time

    svm_linear_accuracy = accuracy_score(
        y_test,
        y_pred_svm_linear
    )

    svm_linear_accuracies.append(svm_linear_accuracy)
    svm_linear_times.append(svm_linear_time)

    print("SVM Linear Accuracy:", svm_linear_accuracy)
    print("SVM Linear Time:", svm_linear_time)
    return svm_linear_accuracies,svm_linear_times





def svm_rbf_algorithm(X_train_encoded,X_test_encoded,y_train,y_test,kernel="rbf",C=1.0,gamma="scale"):
    svm_rbf_accuracies=[]
    svm_rbf_times=[]

    svm_rbf = SVC(kernel=kernel,C=C,gamma=gamma)

    start_time = time.time()

    svm_rbf.fit(X_train_encoded, y_train)

    y_pred_svm_rbf = svm_rbf.predict(X_test_encoded)

    end_time = time.time()

    svm_rbf_time = end_time - start_time

    svm_rbf_accuracy = accuracy_score(
        y_test,
        y_pred_svm_rbf
    )

    svm_rbf_accuracies.append(svm_rbf_accuracy)
    svm_rbf_times.append(svm_rbf_time)

    print("SVM RBF Accuracy:", svm_rbf_accuracy)
    print("SVM RBF Time:", svm_rbf_time)

    return svm_rbf_accuracies,svm_rbf_times





def RF_algorithm(X_train_encoded,X_test_encoded,y_train,y_test,n_estimators=300,criterion="gini",max_depth=None,max_features="sqrt",random_state=42,n_jobs=-1):
    RF_accuracies=[]

    rf = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,max_features=max_features,random_state=random_state,n_jobs=n_jobs)

    rf.fit(X_train_encoded, y_train)
    y_pred_RF = rf.predict(X_test_encoded)

    RF_accuracy = accuracy_score(y_test, y_pred_RF)

    RF_accuracies.append(RF_accuracy)
    print(f"RF Accuracy: {RF_accuracy}")

    return RF_accuracies