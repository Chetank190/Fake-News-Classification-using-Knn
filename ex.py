import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def process_data(data, labels):
    """
	Preprocess a dataset of strings into vector representations.

    Parameters
    ----------
    	data: numpy array
    		An array of N strings.
    	labels: numpy array
    		An array of N integer labels.

    Returns
    -------
    train_X: numpy array
		Array with shape (N, D) of N inputs.
    train_Y:
    	Array with shape (N,) of N labels.
    val_X:
		Array with shape (M, D) of M inputs.
    val_Y:
    	Array with shape (M,) of M labels.
    test_X:
		Array with shape (M, D) of M inputs.
    test_Y:
    	Array with shape (M,) of M labels.
	"""

    # Split the dataset of string into train, validation, and test
    # Use a 70/15/15 split
    # train_test_split shuffles the data before splitting it
    # Stratify keeps the proportion of labels the same in each split

    # -- WRITE THE SPLITTING CODE HERE --

    train_X, test_X_split, train_Y, test_Y_split = train_test_split(data, labels, test_size=0.30, stratify=labels,
                                                                    random_state=20)
    test_X, val_X, test_Y, val_Y = train_test_split(test_X_split, test_Y_split, test_size=0.50, stratify=test_Y_split,
                                                    random_state=20)

    # Preprocess each dataset of strings into a dataset of feature vectors
    # using the CountVectorizer function.
    # Note, fit the Vectorizer using the training set only, and then
    # transform the validation and test sets.

    # -- WRITE THE PROCESSING CODE HERE --
    vector = CountVectorizer(ngram_range=(1, 3))

    train_X = vector.fit_transform(train_X)
    # print(vector.vocabulary_)
    # print(train_X.shape)
    # print(vector.stop_words_)
    test_X = vector.transform(test_X)
    val_X = vector.transform(val_X)

    # print(vector.get_feature_names())
    # print(train_X)
    # Return the training, validation, and test set inputs and labels

    # -- RETURN THE ARRAYS HERE --
    return train_X, train_Y, val_X, val_Y, test_X, test_Y


def select_knn_model(train_X, val_X, train_Y, val_Y):
    """
	Test k in {1, ..., 20} and return the a k-NN model
	fitted to the training set with the best validation loss.

    Parameters
    ----------
    	train_X: numpy array
    		Array with shape (N, D) of N inputs.
    	train_X: numpy array
    		Array with shape (M, D) of M inputs.
    	train_Y: numpy array
    		Array with shape (N,) of N labels.
    	val_Y: numpy array
    		Array with shape (M,) of M labels.

    Returns
    -------
    best_model : KNeighborsClassifier
    	The best k-NN classifier fit on the training data
    	and selected according to validation loss.
  	best_k : int
    	The best k value according to validation loss.
	"""
    acc = []
    model = []
    for k in range(1, 21):
        knn1 = KNeighborsClassifier(n_neighbors=k)
        test = knn1.fit(train_X, train_Y)
        predict_v = test.predict(val_X)
        acc.append(metrics.accuracy_score(val_Y, predict_v))
        model.append(knn1)
    print("Max Acc : ", max(acc), "index : ", acc.index(max(acc))+1)
    print("Acc : ", acc)
    print("model: ", model, '\n',  "Model: ",model[acc.index(max(acc))])

    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, 21), acc, color='blue', linestyle='dashed',
    #          marker='o', markerfacecolor='red', markersize=10)
    #
    # plt.title('accuracy vs. K Value')
    # plt.xlabel('K')
    # plt.ylabel('Accuracy')
    # plt.show()
    # print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))
    return model[acc.index(max(acc))], acc.index(max(acc)) + 1


np.random.seed(3142021)
random.seed(3142021)


def load_data():
    # Load the data
    with open('./clean_fake.txt', 'r') as f:
        fake = [l.strip() for l in f.readlines()]
    with open('./clean_real.txt', 'r') as f:
        real = [l.strip() for l in f.readlines()]

    # Each element is a string, corresponding to a headline
    data = np.array(real + fake)
    labels = np.array([0] * len(real) + [1] * len(fake))
    return data, labels


def main():
    data, labels = load_data()
    # print(len(data))
    # print(len(labels))
    train_X, train_Y, val_X, val_Y, test_X, test_Y = process_data(data, labels)
    # print("test y",len(test_Y))
    # #print("test x", len(test_X))
    # #print("train x",len(train_X))
    # print("train y", len(train_Y))
    # #print("val_x",len(val_X))
    # print("val_y",len(val_Y))
    # print("val_y", train_Y)
    best_model, best_k = select_knn_model(train_X, val_X, train_Y, val_Y)
    print("best_model", best_model)
    print("best_k", best_k)
    test_accuracy = best_model.score(test_X, test_Y)
    print("Selected K: {}".format(best_k))
    print("Test Acc: {}".format(test_accuracy))


if __name__ == '__main__':
    main()
