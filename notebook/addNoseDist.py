import os
import pandas as pd
import math

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def hateThis(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                leftWrist_x_distance = []
                leftWrist_y_distance = []
                rightWrist_x_distance = []
                rightWrist_y_distance = []
                # leftEyeDistance = []
                # rightEyeDistance = []
                # leftEarDistance = []
                # rightEarDistance = []
                # leftShoulderDistance = []
                # rightShoulderDistance = []
                # leftElbowDistance = []
                # rightElbowDistance = []
                # leftWristDistance = []
                # rightWristDistance = []
                # leftHipDistance = []
                # rightHipDistance = []
                # leftKneeDistance = []
                # rightKneeDistance = []
                # leftAnkleDistance = []
                # rightAnkleDistance = []
                fileName = os.path.join(directory, file)
                print(fileName)
                data = pd.read_csv(fileName)
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
                data.to_csv(fileName, index=0)

                    # nose = [data.iloc[i]['nose_x'], data.iloc[i]['nose_y']]
                    # lefteye = [data.iloc[i]['leftEye_x'], data.iloc[i]['leftEye_y']]
                    # righteye = [data.iloc[i]['rightEye_x'], data.iloc[i]['rightEye_y']]
                    # leftear = [data.iloc[i]['leftEar_x'], data.iloc[i]['leftEar_y']]
                    # rightear = [data.iloc[i]['rightEar_x'], data.iloc[i]['rightEar_y']]
                    # leftshoulder = [data.iloc[i]['leftShoulder_x'], data.iloc[i]['leftShoulder_y']]
                    # rightshoulder = [data.iloc[i]['rightShoulder_x'], data.iloc[i]['rightShoulder_y']]
                    # leftelbow = [data.iloc[i]['leftElbow_x'], data.iloc[i]['leftElbow_y']]
                    # rightelbow = [data.iloc[i]['rightElbow_x'], data.iloc[i]['rightElbow_y']]
                    # leftwrist = [data.iloc[i]['leftWrist_x'], data.iloc[i]['leftWrist_y']]
                    # rightwrist = [data.iloc[i]['rightWrist_x'], data.iloc[i]['rightWrist_y']]
                    # lefthip = [data.iloc[i]['leftHip_x'], data.iloc[i]['leftHip_y']]
                    # righthip = [data.iloc[i]['rightHip_x'], data.iloc[i]['rightHip_y']]
                    # leftknee = [data.iloc[i]['leftKnee_x'], data.iloc[i]['leftKnee_y']]
                    # rightknee = [data.iloc[i]['rightKnee_x'], data.iloc[i]['rightKnee_y']]
                    # leftankle = [data.iloc[i]['leftAnkle_x'], data.iloc[i]['leftAnkle_y']]
                    # rightankle = [data.iloc[i]['rightAnkle_x'], data.iloc[i]['rightAnkle_y']]

                    # leftEyeDistance.append(distance(nose, lefteye))
                    # rightEyeDistance.append(distance(nose, righteye))
                    # leftEarDistance.append(distance(nose, leftear))
                    # rightEarDistance.append(distance(nose, rightear))
                    # leftShoulderDistance.append(distance(nose, leftshoulder))
                    # rightShoulderDistance.append(distance(nose, rightshoulder))
                    # leftElbowDistance.append(distance(nose, leftelbow))
                    # rightElbowDistance.append(distance(nose, rightelbow))
                    # leftWristDistance.append(distance(nose, leftwrist))
                    # rightWristDistance.append(distance(nose, rightwrist))
                    # leftHipDistance.append(distance(nose, lefthip))
                    # rightHipDistance.append(distance(nose, righthip))
                    # leftKneeDistance.append(distance(nose, leftknee))
                    # rightKneeDistance.append(distance(nose, rightknee))
                    # leftAnkleDistance.append(distance(nose, leftankle))
                    # rightAnkleDistance.append(distance(nose, rightankle))
                # data['leftEyeDist'] = leftEyeDistance
                # data['rightEyeDist'] = rightEyeDistance
                # data['leftEarDist'] = leftEarDistance
                # data['rightEarDist'] = rightEarDistance
                # data['leftShoulderDist'] = leftShoulderDistance
                # data['rightShoulderDist'] = rightShoulderDistance
                # data['leftElbowDist'] = leftElbowDistance
                # data['rightElbowDist'] = rightElbowDistance
                # data['leftWristDist'] = leftWristDistance
                # data['rightWristDist'] = rightWristDistance
                # data['leftHipDist'] = leftHipDistance
                # data['rightHipDist'] = rightHipDistance
                # data['leftKneeDist'] = leftKneeDistance
                # data['rightKneeDist'] = rightKneeDistance
                # data['leftAnkleDist'] = leftAnkleDistance
                # data['rightAnkleDist'] = rightAnkleDistance
                #print(data)


hateThis("/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/TEST/buy")
hateThis("/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/TEST/communicate")
hateThis("/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/TEST/fun")
hateThis("/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/TEST/hope")
hateThis("/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/TEST/mother")
hateThis("/Users/julialiu/Documents/Semester9/Mobile/Assignment2/assignment_2/TEST/really")
# hateThis("C:\\Users\\andre\Desktop\CSV\\testing")
