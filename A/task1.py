import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def flatten_data(train_data, test_data, val_data, n_component=0.9):
    """
    Flatten the image data to 1D for linear models an use PCA for dimensionality reduction. 
    The [number of imfages, 28, 28] image data are reshaped to [number of imfages, 28*28], 
    and PCA is used to reduce the dimension of 784. When n_component>=1, it is the number of
    the dimension. While 0<n_component<1, it means the percentage boundary of variance.
    
    Args:
        train_data(numpy array): train data of 2D grayscale image (num_samples, height, width)
        val_data(numpy array): validation data of 2D grayscale image (num_samples, height, width)
        test_data(numpy array): test data of 2D grayscale image (num_samples, height, width)
        n_component(int, float) default=0.9: PCA n_component input
    
    Returns:
        train_flattened(numpy array): train data in shape of (num_samples, num_features)
        val_flattened(numpy array): validation data in shape of (num_samples, num_features)
        test_flattened(numpy array): test data in shape of (num_samples, num_features)
    """

    train_flattened = train_data.reshape(train_data.shape[0], -1)
    pca = PCA(n_components=n_component)
    train_flattened = pca.fit_transform(train_flattened)

    val_flattened = val_data.reshape(val_data.shape[0], -1)
    val_flattened = pca.transform(val_flattened)

    test_flattened = test_data.reshape(test_data.shape[0], -1)
    test_flattened = pca.transform(test_flattened)

    return train_flattened, val_flattened, test_flattened

def logistic_regression(train_data,train_label,test_data):
    """
    Implement logistic regression training on train_flattened data and train_label.
    Predict pred_label based on test_flattened and return it

    Args: 
        train_data(numpy array): Data for train
        train_label(numpy array): The train labels
        test_data(numpy array): Data for test

    Returns:
        pred_label(numpy array): Predicted labels for test data
    """
    logreg = sklearn.linear_model.LogisticRegression(solver='lbfgs')
    logreg.fit(train_data,train_label)
    pred_label = logreg.predict(test_data)
    return pred_label

def randomforestclassify(train_data,train_label,test_data):
    """
    Implement randomforest training on train_flattened data and train_label.
    Predict pred_label based on test_flattened and return it

    Args: 
        train_data(numpy array): Data for train
        train_label(numpy array): The train labels
        test_data(numpy array): Data for test

    Returns:
        pred_label(numpy array): Predicted labels for test data
    """
    randf = RandomForestClassifier(n_estimators=100)
    randf.fit(train_data,train_label)
    pred_label = randf.predict(test_data)
    return pred_label

def add_features(original_data,flattened_data):
    """
    Add statistical features (mean and STD) of the original data
    to the flattened data 

    Args:
        original_data(numpy array): The original non-flattened data
        flattened_data(numpy array): The flattened and dimension-reduced data

    Returns:
        new_data(numpy array): the flattened_data combine with mean and STD features
        for each sample(image)
    """
    new_data=[]
    original_data = original_data.reshape(original_data.shape[0], -1)
    for data1, data2 in zip(original_data,flattened_data):
        new_feature = np.concatenate((data2,[np.mean(data1),np.std(data1)]))
        new_data.append(new_feature)
    return np.array(new_data)