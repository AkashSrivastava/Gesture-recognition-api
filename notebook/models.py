import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from pandas.io.json import json_normalize

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)

def most_common(predictions):
    count_dict = {"BUY":0, "COMMUNICATE":0, "FUN":0, "HOPE":0, "MOTHER":0, "REALLY":0}
    for x in predictions:
        count_dict[x] += 1

    #print count_dict
    return max(count_dict, key=count_dict.get)

def flatten_document(y):
    result = {}

    def helper(doc, name=''):
        if type(doc) is dict:
            for dict_name in doc:
                helper(doc[dict_name], name + dict_name + '_')
        elif type(doc) is list:
            for i,a in enumerate(doc):
                helper(a, name + str(i) + '_')
        else:
            result[name[:-1]] = doc
    helper(y)
    return result


def preprocessing(data):
    print("I am in knnmodel")

    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']

    csv_data = np.zeros((len(data), len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(data[i]['score'])
        for obj in data[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)
    data = pd.DataFrame(csv_data, columns=columns)

    leftWrist_x_distance = []
    leftWrist_y_distance = []
    rightWrist_x_distance = []
    rightWrist_y_distance = []
    for i in range(data.shape[0]):
        nose_x = data.iloc[i]['nose_x']
        nose_y = data.iloc[i]['nose_y']

        leftWristXDistance = data.iloc[i]['leftWrist_x']
        leftWristYDistance = data.iloc[i]['leftWrist_y']
        rightWristXDistance = data.iloc[i]['rightWrist_x']
        rightWristYDistance = data.iloc[i]['rightWrist_y']

        leftWrist_x_distance.append(leftWristXDistance - nose_x)
        leftWrist_y_distance.append(leftWristYDistance - nose_y)
        rightWrist_x_distance.append(rightWristXDistance - nose_x)
        rightWrist_y_distance.append(rightWristYDistance - nose_y)
    data['leftWrist_x_dist'] = leftWrist_x_distance
    data['leftWrist_y_dist'] = leftWrist_y_distance
    data['rightWrist_x_dist'] = rightWrist_x_distance
    data['rightWrist_y_dist'] = rightWrist_y_distance

    print(data)

    prediction_array = []
    prediction_array.append(KNNTest(data))
    prediction_array.append(DTTest(data))
    prediction_array.append(SVMTest(data))
    prediction_array.append(RFTest(data))

    print prediction_array
    return prediction_array

def runKNN():
    print("KNN")
    dataset = pd.read_csv("datasubset.csv")

    X=dataset.iloc[:,0:len(dataset.columns)-1]
    print(X)
    print(len(dataset.columns)-1)
    y=dataset.iloc[:,len(dataset.columns)-1]
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

    scaler = StandardScaler()
    scaler.fit(X_train)

    #unscaled training/test data
    X_train.to_csv("train.csv")

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    #save the model as a PICKLE :)
    knn_pickle_filename = 'knn_classifier.pkl'
    knn_pickle = open(knn_pickle_filename, 'wb')
    pickle.dump(classifier, knn_pickle)
    knn_pickle.close()

    y_pred = classifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix,precision_recall_fscore_support
    #print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(precision_recall_fscore_support(y_test,y_pred,average='macro'))

def KNNTest(keypoints):
    train_data = pd.read_csv("/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/train.csv", index_col=0)
    scaler = StandardScaler()
    scaler.fit(train_data)

    # keep_col=['leftElbow_x','leftElbow_y','rightElbow_x','rightElbow_y','leftWrist_x','leftWrist_y','rightWrist_x',
    #           'rightWrist_y']
    keep_col=['leftWrist_x_dist','leftWrist_y_dist','rightWrist_x_dist', 'rightWrist_y_dist']
    keypoints = keypoints[keep_col]

    keypoints = scaler.transform(keypoints)

    knn_pickle = open('/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/knn_classifier.pkl', 'rb')
    knn_model = pickle.load(knn_pickle)
    return most_common(knn_model.predict(keypoints))


def runnDT():

    print("DECISION TREE")
    dataset = pd.read_csv("datasubset.csv")

    X = dataset.iloc[:, 0:len(dataset.columns) - 1]
    print(X)
    y = dataset.iloc[:, len(dataset.columns) - 1]
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
    clf=tree.DecisionTreeClassifier(max_depth=8)
    # lb = LabelEncoder()
    # y_train = lb.fit_transform(y_train)
    # y_test = lb.fit_transform(y_test)


    clf=clf.fit(X,y)
    y_pred=clf.predict(X_test)

    #save the model as a PICKLE :)
    dt_pickle_filename = 'dt_classifier.pkl'
    dt_pickle = open(dt_pickle_filename, 'wb')
    pickle.dump(clf, dt_pickle)
    dt_pickle.close()

    #
    # from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
    # # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(precision_recall_fscore_support(y_test, y_pred, average='macro'))

def DTTest(keypoints):
    lb = LabelEncoder()
    lb.fit(["BUY", "COMMUNICATE", "FUN", "HOPE", "MOTHER", "REALLY"])

    # keep_col=['leftElbow_x','leftElbow_y','rightElbow_x','rightElbow_y','leftWrist_x','leftWrist_y','rightWrist_x',
    #           'rightWrist_y']
    keep_col = ['leftWrist_x_dist', 'leftWrist_y_dist', 'rightWrist_x_dist', 'rightWrist_y_dist']
    keypoints = keypoints[keep_col]

    dt_pickle = open('/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/dt_classifier.pkl', 'rb')
    dt_model = pickle.load(dt_pickle)
    return most_common(dt_model.predict(keypoints))

def runSVM():
    print("SVM")
    dataset = pd.read_csv("datasubset.csv")

    X = dataset.iloc[:, 0:len(dataset.columns) - 1]
    y = dataset.iloc[:, len(dataset.columns) - 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    clf.fit(X_train, y_train)

    #save the model as a PICKLE :)
    svm_pickle_filename = 'svm_classifier.pkl'
    svm_pickle = open(svm_pickle_filename, 'wb')
    pickle.dump(clf, svm_pickle)
    svm_pickle.close()

    y_pred = clf.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
    # print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(precision_recall_fscore_support(y_test, y_pred, average='macro'))

def SVMTest(keypoints):
    # keep_col = ['leftElbow_x', 'leftElbow_y', 'rightElbow_x', 'rightElbow_y', 'leftWrist_x', 'leftWrist_y',
    #             'rightWrist_x','rightWrist_y']
    keep_col = ['leftWrist_x_dist', 'leftWrist_y_dist', 'rightWrist_x_dist', 'rightWrist_y_dist']
    keypoints = keypoints[keep_col]

    svm_pickle = open('/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/svm_classifier.pkl', 'rb')
    svm_model = pickle.load(svm_pickle)
    return most_common(svm_model.predict(keypoints))

def runRF():
    print("RF")
    dataset = pd.read_csv("datasubset.csv")

    X = dataset.iloc[:, 0:len(dataset.columns) - 1]
    y = dataset.iloc[:, len(dataset.columns) - 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

    clf = RandomForestClassifier(n_estimators=100, max_depth=8,random_state = 0)
    clf.fit(X, y)

    #save the model as a PICKLE :)
    rf_pickle_filename = 'rf_classifier.pkl'
    rf_pickle = open(rf_pickle_filename, 'wb')
    pickle.dump(clf, rf_pickle)
    rf_pickle.close()

    y_pred=clf.predict(X_test)

    # from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
    # print(classification_report(y_test, y_pred))
    # print(precision_recall_fscore_support(y_test, y_pred, average='macro'))

def RFTest(keypoints):
    # keep_col = ['leftElbow_x', 'leftElbow_y', 'rightElbow_x', 'rightElbow_y', 'leftWrist_x', 'leftWrist_y',
    #             'rightWrist_x','rightWrist_y']
    keep_col = ['leftWrist_x_dist', 'leftWrist_y_dist', 'rightWrist_x_dist', 'rightWrist_y_dist']
    keypoints = keypoints[keep_col]

    rf_pickle = open('/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/rf_classifier.pkl', 'rb')
    rf_model = pickle.load(rf_pickle)
    return most_common(rf_model.predict(keypoints))

def cleanData():
    print ("Cleaning Data");
    dataset = pd.read_csv("dataset.csv")
    # keep_col=['leftElbow_x','leftElbow_y','rightElbow_x','rightElbow_y','leftWrist_x','leftWrist_y','rightWrist_x',
    #           'rightWrist_y', 'class']
    keep_col = ['leftWrist_x_dist', 'leftWrist_y_dist', 'rightWrist_x_dist','rightWrist_y_dist', 'class']
    datasubset = dataset[keep_col]
    datasubset.to_csv("datasubset.csv", index=False)

if __name__ == '__main__':
    cleanData()
    #runKNN()
    runnDT()
    #runSVM()
    runRF()
