import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./A')
import A.task1

class Args: None

def taskA():
    data = np.load('./Datasets/breastmnist.npz')

    # assign data to train, validation, and test sets. Normalize the pixel value. 
    train_data = data['train_images']/255 
    train_label = data['train_labels']
    val_data = data['val_images']/255
    val_label = data['val_labels']
    test_data = data['test_images']/255
    test_label = data['test_labels']

    n_components = np.arange(0.05, 1, 0.05)
    logistic_accuracy = []
    randomforest_accuracy1 = []
    randomforest_accuracy2 = []

    for n_component in n_components:
        train_flattened, val_flattened, test_flattened = A.task1.flatten_data(train_data, test_data, val_data, n_component)

        pred_label = A.task1.logistic_regression(train_flattened,train_label,test_flattened)
        accuracy = A.task1.sklearn.metrics.accuracy_score(test_label, pred_label)
        print('Accuracy on logistic test: '+str(accuracy))
        logistic_accuracy.append(accuracy)

        pred_label1 = A.task1.randomforestclassify(train_flattened,train_label,test_flattened)
        accuracy1 = A.task1.sklearn.metrics.accuracy_score(test_label, pred_label1)
        print('Accuracy on randomforest test: '+str(accuracy1))
        randomforest_accuracy1.append(accuracy1)

        pred_label2 = A.task1.randomforestclassify(A.task1.add_features(train_data,train_flattened),train_label,A.task1.add_features(test_data,test_flattened))
        accuracy2 = A.task1.sklearn.metrics.accuracy_score(test_label, pred_label2)
        print('Accuracy on randomforest test with more features: '+str(accuracy2))
        randomforest_accuracy2.append(accuracy2)

    plt.plot(n_components,logistic_accuracy,'o-',label='logistic regression accuracy')
    plt.plot(n_components,randomforest_accuracy1,'o-',label='randomforest accuracy')
    plt.plot(n_components,randomforest_accuracy2,'o-',label='randomforest with more features accuracy')
    plt.xlabel("n_component for PCA, percentage of variance (decision boundary for number of features)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Task A")
    plt.savefig('TaskA.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    args: None
    taskA()
