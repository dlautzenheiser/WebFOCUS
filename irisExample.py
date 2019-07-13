#
# core from https://machinelearningmastery.com/machine-learning-in-python-step-by-step/ 
# Jason Brownlee, article published on 10 June 2016, updated in March 2017 
#

# for python system input arguments 
import sys

import pandas
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import numpy



# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
dataset = pandas.read_csv(url, names=names)
print('\nDownloaded Iris data from University of California, Irvine...')


def describe_iris():
    print('\nDataset information:')
    print(dataset.info())
    # print(dataset.dtypes)
    print('\nDataset shape (rows,columns):', dataset.shape)
    print('\nDataset first 10 records:')
    print(dataset.head(10))
    print('\nDataset last 10 records:')
    print(dataset.tail(10))
    print('\nDataset species distribution:')
    print(dataset.groupby('species').size())
    print('\nDataset statistics by species:')
    print(dataset.describe())



def graph_iris():
    import seaborn as sns
    sns.set(style="ticks", color_codes=True)
    #iris = sns.load_dataset("iris")

    # box and whisker plots
    #dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    #plt.show()
    sns.boxplot(data=dataset, orient="v", palette="BuGn_r"); plt.show()

    # histograms
    #dataset.hist()
    #plt.show()

    # scatter plot matrix
    #scatter_matrix(dataset)
    #plt.show()
    sns.pairplot(dataset, hue="species", palette="husl",markers=["o","s","D"], diag_kind="hist"); plt.show()



def model_iris():
    # Load libraries
    from sklearn import model_selection
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC



    # Split-out validation dataset
    array = dataset.values
    X = array[:, 0:4]
    y = array[:, 4]
    validation_size = 0.20
    seed = 7
    scoring = 'accuracy'
    X_train, X_validation, y_train, y_validation = \
        model_selection.train_test_split(X, y, test_size=validation_size,random_state=seed)

    # compare the algorithms
    algorithms = []
    algorithms.append(('Logistic Regresion(LR)', LogisticRegression()))
    algorithms.append(('Linear Discriminant Analysis (LDA)', LinearDiscriminantAnalysis()))
    algorithms.append(('K-Nearest Neighbors (KNN)', KNeighborsClassifier()))
    algorithms.append(('Classification and Regression Trees (CART)', DecisionTreeClassifier()))
    algorithms.append(('Gaussian Naive Bayes (NB)', GaussianNB()))
    algorithms.append(('Support Vector Classification (SVC)', SVC()))
    algorithms.append(('Linear SVC', LinearSVC()))


    if sys.argv[1] == 'model':
        # evaluate each model in turn
        results = []
        names = []
        print('\nIris Predictive Model Accuracy:')
        for description, model in algorithms:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(description)
            message = "   mean=%f stddev=%f: %s" % (cv_results.mean(), cv_results.std(), description)
            print(message)

        # Compare Algorithms visually
        #fig = plt.figure()
        #fig.suptitle('Algorithm Comparison')
        #ax = fig.add_subplot(111)
        #plt.boxplot(results)
        #ax.set_xticklabels(names)
        #plt.show()

    elif sys.argv[1] == 'validate':
        print('\nRunning KNN predictive model on validation data:')
        # Make predictions on validation dataset
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_validation)

        accuracy = (accuracy_score(y_validation, predictions) * 100)
        print('\nConfusion Matrix (accuracy is', accuracy, '%):')
        #print(confusion_matrix(y_validation, predictions))
        confusionarray = confusion_matrix(y_validation, predictions)
        print('A:setosa    \t', confusionarray[0,0], '\t',confusionarray[0,1],'\t',confusionarray[0,2])
        print('A:versicolor\t', confusionarray[1,0], '\t',confusionarray[1,1],'\t',confusionarray[1,2])
        print('A:virginia  \t', confusionarray[2,0], '\t',confusionarray[2,1],'\t',confusionarray[2,2])
        print('\t\tP:set \tP:ver \tP:vir')

        print('\nClasification Report:')
        print(classification_report(y_validation, predictions))
    else:
        show_usage()



# function to show arguments
def show_arguments():
    print('Running Python script named %s' % (sys.argv[0]))
    if len(sys.argv) > 1:
        # in addition to program name, at least one parameter was provided
        print('#Arguments= ', len(sys.argv))
        print('Arguments= ', str(sys.argv))
        count = 0
        while (count < len(sys.argv)):
            print("Argument#%d is %s" % (count, sys.argv[count]))
            count = count + 1

        NbrVariables = len(sys.argv)
        print('Setting NbrVariables= ', NbrVariables)

    else:
        # program was run without any arguments
        print('No program arguments were provided.')
        NbrVariables = 0

    return NbrVariables


def show_usage():
    print('\nUsage:')
    print('  To see this, enter: ')
    print('     $ python irisexample.py help')
    print('  To see information about Iris dataset, enter: ')
    print('     $ python irisexample.py describe')
    print('  To graph information about Iris dataset, enter: ')
    print('     $ python irisexample.py graph')
    print('  To see information about Iris predictive models, enter: ')
    print('     $ python irisexample.py model')
    print('  To validate Iris KNN predictive model, enter: ')
    print('     $ python irisexample.py validate')


# main function
def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'debug':
            show_arguments()
        elif sys.argv[1] == 'describe':
            describe_iris()
        elif sys.argv[1] == 'graph':
            graph_iris()

        elif sys.argv[1] == 'model' or sys.argv[1] == 'validate':
            model_iris()
        else:
            show_usage()
    else:
        show_usage()


# call main function
main()
