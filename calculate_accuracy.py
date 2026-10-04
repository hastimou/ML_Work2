import numpy as np
from sklearn.metrics import accuracy_score,classification_report


def calculate_accuracy_time(RF_accuracies,svm_linear_accuracies,svm_rbf_accuracies,svm_linear_times,svm_rbf_times):
    mean_RF_accuracy = np.mean(RF_accuracies)
    std_RF_accuracy = np.std(RF_accuracies)

    mean_svm_linear_accuracy = np.mean(
        svm_linear_accuracies
    )

    std_svm_linear_accuracy = np.std(
        svm_linear_accuracies
    )

    mean_svm_rbf_accuracy = np.mean(
        svm_rbf_accuracies
    )

    std_svm_rbf_accuracy = np.std(
        svm_rbf_accuracies
    )

    mean_svm_linear_time = np.mean(
        svm_linear_times
    )

    mean_svm_rbf_time = np.mean(
        svm_rbf_times
    )

    return mean_RF_accuracy, std_RF_accuracy,mean_svm_linear_accuracy,std_svm_linear_accuracy,mean_svm_rbf_accuracy,std_svm_rbf_accuracy,mean_svm_linear_time,mean_svm_rbf_time






def show_results(mean_RF_accuracy,std_RF_accuracy,mean_svm_linear_accuracy,std_svm_linear_accuracy,mean_svm_linear_time,mean_svm_rbf_accuracy,std_svm_rbf_accuracy,mean_svm_rbf_time):
    print("\n" + "=" * 60)
    print("SVM/RF RESULTS")
    print("=" * 60)

    print(
        f"RF Mean Accuracy: "
        f"{mean_RF_accuracy:.4f}"
    )

    print(
        f"RF Accuracy (%): "
        f"{mean_RF_accuracy * 100:.2f}%"
    )
    print(
        f"RF Std Accuracy: "
        f"{std_RF_accuracy:.4f}"
    )
    print()

    print(
        f"Linear SVM Mean Accuracy: "
        f"{mean_svm_linear_accuracy:.4f}"
    )

    print(
        f"Linear SVM Accuracy (%): "
        f"{mean_svm_linear_accuracy * 100:.2f}%"
    )

    print(
        f"Linear SVM Std: "
        f"{std_svm_linear_accuracy:.4f}"
    )

    print(
        f"Linear SVM Mean Time: "
        f"{mean_svm_linear_time:.6f} seconds"
    )

    print()

    print(
        f"RBF SVM Mean Accuracy: "
        f"{mean_svm_rbf_accuracy:.4f}"
    )

    print(
        f"RBF SVM Accuracy (%): "
        f"{mean_svm_rbf_accuracy * 100:.2f}%"
    )

    print(
        f"RBF SVM Std: "
        f"{std_svm_rbf_accuracy:.4f}"
    )

    print(
        f"RBF SVM Mean Time: "
        f"{mean_svm_rbf_time:.6f} seconds"
    )