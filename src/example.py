import os
import numpy as np
from bayesclassifier import BayesClassifier

data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
print(data_folder)


def load_data(input_path, type_):
    # D1=pd.read_csv(input_path,sep=',')
    # print D1
    D = np.loadtxt(input_path, delimiter=',', dtype=type_)
    return D


xtrain_path = os.path.join(data_folder, 'x_train.csv')
ytrain_path = os.path.join(data_folder, 'y_train.csv')
xtest_path = os.path.join(data_folder, 'x_test.csv')

out_path = os.path.join(data_folder, 'probs_test.csv')

xtrain = load_data(xtrain_path, float)
ytrain = load_data(ytrain_path, int)
xtest = load_data(xtest_path, float)

classifier = BayesClassifier(xtrain, ytrain)
P = classifier.classify(xtrain)

np.savetxt(out_path, P, delimiter=",")  # write output to file

print(classifier)
