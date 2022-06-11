# Fake-News-Classification-using-Knn

Classification with Nearest Neighbours. In this question, you will use the
scikit-learn’s KNN classifer to classify real vs. fake news headlines. The aim of this question
is for you to read the scikit-learn API and get comfortable with training/validation
splits.  
We will use a dataset of 1298 “fake news” headlines (which mostly include headlines of articles
classified as biased, etc.) and 1968 “real” news headlines, where the “fake news” headlines
are from https://www.kaggle.com/mrisdal/fake-news/data and “real news” headlines are
from https://www.kaggle.com/therohk/million-headlines. The data were cleaned by
removing words from titles not part of the headlines, removing special characters and restricting
real news headlines after October 2016 using the word ”trump”. The cleaned data
and starter code are available as clean_real.txt and clean_fake.txt in hw1_starter.zip
on the course webpage. It is expected that you use these cleaned data sources for this assignment.
We are providing starter code for this assignment. To run the code you simply need to to
run hw1.py using your favourite Python interpreter. To make the code correct, you will need
to fill in the body of two functions. If you do this correctly, the code should run and output
something that looks like the following:  
Selected K: 10  
Test Acc: 0.5  
Before you implement anything, or if you implement it incorrectly, the code may raise an
Exception. This is expected behaviour.  
You will build a KNN classifier to classify real vs. fake news headlines. Instead of coding the
KNN yourself, you will do what we normally do in practice — use an existing implementation.
You should use the KNeighborsClassifier included in sklearn. Note that figuring out
1https://www.cs.toronto.edu/~cmaddis/courses/sta314_f21/sta314_f21_syllabus.pdf

(a) Complete the function process_data. It should do the following.
• First, split the entire dataset randomly into 70% training, 15% validation, and
15% test examples using train_test_split function (https://scikit-learn.
org/stable/modules/generated/sklearn.model_selection.train_test_split.
html). You can use the stratify option to keep the label proportions the same in the
split.  
• Then, preprocess the data using a CountVectorizer (https://scikit-learn.org/
stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.
html#sklearn.feature_extraction.text.CountVectorizer). You will need to
understand what this preprocessing does, so you should play around with it. Also,
the CountVectorizer should be fit only on the training set.
In your writeup, report the function process_data that you wrote.  
(b) Complete the function select_knn_model that selects a k-NN classifer using a
training set and a validation set to classify between real vs. fake news. This function
should do the following.  
• Iterate over k values between 1 to 20.  
• For each k value, fit a KNeighborsClassifier to the training set, leaving other
arguments at their default values.  
• Measure the validation accuracy.  
• Return the best choice of k and the corresponding model that has been fit to the
training set with that value of k.  
In your writeup, report the function select_knn_model that you wrote, as well as the
output of the hw1.py script.  
(c) Repeat part (b), passing argument metric=‘cosine’ to the KNeighborsClassifier.
You should observe an improvement in accuracy. How does metric=‘cosine’ compute
the distance between data points, and why might this perform better than the Euclidean
metric (default) here? Hint: consider the dataset [‘cat’, ‘bulldozer’, ‘cat cat cat’].  
