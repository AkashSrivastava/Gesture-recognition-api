import models
import pandas as pd
import os

def sign_to_int(sign):
    if(sign == "BUY"):
        return 0
    if (sign == "COMMUNICATE"):
        return 1
    if (sign == "FUN"):
        return 2
    if (sign == "HOPE"):
        return 3
    if (sign == "MOTHER"):
        return 4
    if (sign == "REALLY"):
        return 5

def print_confusion_matrices(confusion_matrices):
    print("BUY:")
    print("True Negative: {}".format(confusion_matrices[0][0][0]))
    print("False Positive: {}".format(confusion_matrices[0][0][1]))
    print("False Negative: {}".format(confusion_matrices[0][1][0]))
    print("True Positive: {}".format(confusion_matrices[0][1][1]))
    print("\n")
    print("COMMUNICATE:")
    print("True Negative: {}".format(confusion_matrices[1][0][0]))
    print("False Positive: {}".format(confusion_matrices[1][0][1]))
    print("False Negative: {}".format(confusion_matrices[1][1][0]))
    print("True Positive: {}".format(confusion_matrices[1][1][1]))
    print("\n")
    print("FUN:")
    print("True Negative: {}".format(confusion_matrices[2][0][0]))
    print("False Positive: {}".format(confusion_matrices[2][0][1]))
    print("False Negative: {}".format(confusion_matrices[2][1][0]))
    print("True Positive: {}".format(confusion_matrices[2][1][1]))
    print("\n")
    print("HOPE:")
    print("True Negative: {}".format(confusion_matrices[3][0][0]))
    print("False Positive: {}".format(confusion_matrices[3][0][1]))
    print("False Negative: {}".format(confusion_matrices[3][1][0]))
    print("True Positive: {}".format(confusion_matrices[3][1][1]))
    print("\n")
    print("MOTHER:")
    print("True Negative: {}".format(confusion_matrices[4][0][0]))
    print("False Positive: {}".format(confusion_matrices[4][0][1]))
    print("False Negative: {}".format(confusion_matrices[4][1][0]))
    print("True Positive: {}".format(confusion_matrices[4][1][1]))
    print("\n")
    print("REALLY:")
    print("True Negative: {}".format(confusion_matrices[5][0][0]))
    print("False Positive: {}".format(confusion_matrices[5][0][1]))
    print("False Negative: {}".format(confusion_matrices[5][1][0]))
    print("True Positive: {}".format(confusion_matrices[5][1][1]))

def printModelStats(confusion_matrices):
    print("\t\tprecision\trecall\t\tf1-score\taccuracy")
    for x in range(0,6):
        precision = 0
        recall = 0
        f1 = 0
        accuracy = 0

        tp=float(confusion_matrices[x][1][1])
        tn=float(confusion_matrices[x][0][0])
        fp=float(confusion_matrices[x][0][1])
        fn=float(confusion_matrices[x][1][0])
        if tp > 0:
            precision=round(tp/(tp+fp),2)
            recall=round(tp/(tp+fn),2)
            f1=round(precision*recall/(precision+recall),2)
            accuracy=round((tp+tn)/(tp+tn+fn+fp),2)
        print("{0}\t\t{1:.2f}\t\t{2:.2f}\t\t{3:.2f}\t\t{4:.2f}".format(x, precision, recall, f1, accuracy))


if __name__ == '__main__':
    dirs = os.listdir("/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/TEST")
    folder_path = "/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/TEST"

    #instantiate the confusion matrices for each sign
    # 0-BUY
    # 1-COMMUNICATE
    # 2-FUN
    # 3-HOPE
    # 4-MOTHER
    # 5-REALLY

    confusion_matrices=[[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]

    for i in range(len(dirs)):
        #print(dirs[i])
        files = [file_path for file_path in os.listdir(folder_path + "/" + dirs[i]) if file_path.endswith('.csv')]
        for x in range(len(files)):
            #print(files[x])
            keypoints = pd.read_csv(os.path.join(folder_path+"/"+dirs[i], files[x]), )
            knn_prediction = models.KNNTest(keypoints)
            if(knn_prediction==dirs[i].upper()):
                #if the prediction was correct increment true positive and true negative for correct matrices
                confusion_matrices[sign_to_int(dirs[i].upper())][1][1] += 1
                for d in dirs:
                    if d != dirs[i]:
                        confusion_matrices[sign_to_int(d.upper())][0][0] += 1

            else:
                #if the prediction was incorrect increment false negative for the current matrix and increment the false positive for the classfied element
                confusion_matrices[sign_to_int(dirs[i].upper())][1][0] += 1
                confusion_matrices[sign_to_int(knn_prediction)][0][1] += 1
    print("KNN")
    #print_confusion_matrices(confusion_matrices)
    printModelStats(confusion_matrices)

    confusion_matrices=[[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]

    for i in range(len(dirs)):
        #print(dirs[i])
        files = [file_path for file_path in os.listdir(folder_path + "/" + dirs[i]) if file_path.endswith('.csv')]
        for x in range(len(files)):
            #print(files[x])
            keypoints = pd.read_csv(os.path.join(folder_path+"/"+dirs[i], files[x]), )
            #print(files[x])
            predictions = models.DTTest(keypoints)
            #print predictions
            dt_prediction = predictions
            if(dt_prediction==dirs[i].upper()):
                #if the prediction was correct increment true positive and true negative for correct matrices
                confusion_matrices[sign_to_int(dirs[i].upper())][1][1] += 1
                for d in dirs:
                    if d != dirs[i]:
                        confusion_matrices[sign_to_int(d.upper())][0][0] += 1

            else:
                #if the prediction was incorrect increment false negative for the current matrix and increment the false positive for the classfied element
                confusion_matrices[sign_to_int(dirs[i].upper())][1][0] += 1
                confusion_matrices[sign_to_int(dt_prediction)][0][1] += 1
    print("DT")
    #print_confusion_matrices(confusion_matrices)
    printModelStats(confusion_matrices)

    confusion_matrices=[[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]

    for i in range(len(dirs)):
        #print(dirs[i])
        files = [file_path for file_path in os.listdir(folder_path + "/" + dirs[i]) if file_path.endswith('.csv')]
        for x in range(len(files)):
            #print(files[x])
            keypoints = pd.read_csv(os.path.join(folder_path+"/"+dirs[i], files[x]), )
            svm_prediction = models.SVMTest(keypoints)
            if(svm_prediction==dirs[i].upper()):
                #if the prediction was correct increment true positive and true negative for correct matrices
                confusion_matrices[sign_to_int(dirs[i].upper())][1][1] += 1
                for d in dirs:
                    if d != dirs[i]:
                        confusion_matrices[sign_to_int(d.upper())][0][0] += 1

            else:
                #if the prediction was incorrect increment false negative for the current matrix and increment the false positive for the classfied element
                confusion_matrices[sign_to_int(dirs[i].upper())][1][0] += 1
                confusion_matrices[sign_to_int(svm_prediction)][0][1] += 1
    print("SVM")
    #print_confusion_matrices(confusion_matrices)
    printModelStats(confusion_matrices)


    confusion_matrices=[[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]

    for i in range(len(dirs)):
        #print(dirs[i])
        files = [file_path for file_path in os.listdir(folder_path + "/" + dirs[i]) if file_path.endswith('.csv')]
        for x in range(len(files)):
            #print(files[x])
            keypoints = pd.read_csv(os.path.join(folder_path+"/"+dirs[i], files[x]), )
            rf_prediction = models.RFTest(keypoints)
            if(rf_prediction==dirs[i].upper()):
                #if the prediction was correct increment true positive and true negative for correct matrices
                confusion_matrices[sign_to_int(dirs[i].upper())][1][1] += 1
                for d in dirs:
                    if d != dirs[i]:
                        confusion_matrices[sign_to_int(d.upper())][0][0] += 1

            else:
                #if the prediction was incorrect increment false negative for the current matrix and increment the false positive for the classfied element
                confusion_matrices[sign_to_int(dirs[i].upper())][1][0] += 1
                confusion_matrices[sign_to_int(rf_prediction)][0][1] += 1
    print("RF")
    #print_confusion_matrices(confusion_matrices)
    printModelStats(confusion_matrices)



