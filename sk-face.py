from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import numpy as np
import pickle
import cv2
from face_landmark_dnn_vgg_RT import get_landmarks
import dlib
train_data_path = 'D:/DeepLearning/face/train_landmark_vgg/'
train_label_path = 'D:/DeepLearning/face/train_landmark_vgg/'
test_data_path = 'D:/DeepLearning/face/test_landmark_vgg/'
test_label_path = 'D:/DeepLearning/face/test_landmark_vgg/'
trainset_size = 95200
testset_size = 47600
num_feat = 68
n_class = 2
detector = dlib.get_frontal_face_detector()
fname = 'D:/DeepLearning/face/Face-LandMark_with_Dlib/facial-landmarks/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(fname)
l_eye_pct = 0.33
r_eye_pct = 0.66
eyes_level_pct = 0.4
train_indices = np.random.permutation(trainset_size//num_feat)  # generate random training-data index
test_indices = np.random.permutation(testset_size//num_feat)  # generate random testing-data index
classifier = ["Jerry", "Stranger"]
def read_data(dataset=1):
    """
    reading data from dir
    with random
    :param dataset:1 for trainning, 0 for testing
    :return strength, angle, label
    """
    if dataset == 1:
        tmp_output_txt = os.path.join(train_label_path, 'train_face_label.txt')
        tmp_output_txt_writer = open(tmp_output_txt, 'r')
        Label = tmp_output_txt_writer.read()
        df = pd.read_csv(train_data_path + 'trainset.csv')
        strength_tmp = np.array(df['strength'])
        angle_tmp = np.array(df['angle'])
        strength = []
        angle = []
        label = []
        for _ in range(trainset_size):
            if _ % num_feat == 0:
                strength_tmp2 = []
                angle_tmp2 = []
                for __ in range(num_feat):
                    strength_tmp2.append(strength_tmp[__ + _])
                    angle_tmp2.append(angle_tmp[__ + _])
                strength.append(strength_tmp2)
                angle.append(angle_tmp2)
                label.append([int(Label[_ * (n_class + 1)]), int(Label[_ * (n_class + 1) + 1])])
                del strength_tmp2
                del angle_tmp2
        del strength_tmp
        del angle_tmp
        strength = np.array(strength)
        angle = np.array(angle)
        strength2 = np.zeros_like(strength)
        angle2 = np.zeros_like(angle)
        label2 = np.zeros_like(label)
        for i in range(trainset_size // num_feat):
            counter = train_indices[i]
            strength2[i] = strength[counter]
            angle2[i] = angle[counter]
            label2[i] = label[counter]
        tmp_output_txt_writer.flush()
        tmp_output_txt_writer.close()
        del Label
        del label
        del angle
        del strength
        return strength2, angle2 / 180, label2
    elif dataset == 0:
        x = []
        tmp_output_txt = os.path.join(test_data_path, 'test_face_label.txt')
        tmp_output_txt_writer = open(tmp_output_txt, 'r')
        Label = tmp_output_txt_writer.read()
        df = pd.read_csv(test_data_path + 'testset.csv')
        strength_tmp = np.array(df['strength'])
        angle_tmp = np.array(df['angle'])
        strength = []
        angle = []
        label = []
        for _ in range(testset_size):
            if _ % num_feat == 0:
                strength_tmp2 = []
                angle_tmp2 = []
                for __ in range(num_feat):
                    strength_tmp2.append(strength_tmp[__ + _])
                    angle_tmp2.append(angle_tmp[__ + _])
                strength.append(strength_tmp2)
                angle.append(angle_tmp2)
                label.append([int(Label[_ * (n_class + 1)]), int(Label[_ * (n_class + 1) + 1])])
                del strength_tmp2
                del angle_tmp2
        del strength_tmp
        del angle_tmp
        strength = np.array(strength)
        angle = np.array(angle)
        strength2 = np.zeros_like(strength)
        angle2 = np.zeros_like(angle)
        label2 = np.zeros_like(label)
        for i in range(testset_size // num_feat):
            counter = test_indices[i]
            strength2[i] = strength[counter]
            angle2[i] = angle[counter]
            label2[i] = label[counter]
        tmp_output_txt_writer.flush()
        tmp_output_txt_writer.close()
        del Label
        del label
        del angle
        del strength
        return strength2, angle2 / 180, label2
    else:
        print('argument error')

def Maximum(input_array):
    """find max in 1D-array"""
    max_val = input_array[0]
    idx = 0
    for i in range(len(input_array)):
        if input_array[i] > max_val:
            max_val = input_array[i]
            idx = i
    return max_val, idx

def train():
    trainX_strength, trainX_angle, trainY = read_data(1)
    testX_strength, testX_angle, testY = read_data(0)
    model_str = LinearRegression()
    model_angle = LinearRegression()
    model_str.fit(trainX_strength, trainY)
    model_angle.fit(trainX_angle, trainY)
    pred_str = model_str.predict(testX_strength)
    pred_angle = model_angle.predict(testX_angle)
    counter = 0
    for _ in range(len(pred_str)):
        max1, idx1 = Maximum(pred_str[_])
        max2, idx2 = Maximum(testY[_])
        if idx1 == idx2:
            counter += 1
    accuracy_str = counter / len(pred_str)
    print("strength:", accuracy_str)

    counter = 0
    for _ in range(len(pred_angle)):
        max1, idx1 = Maximum(pred_angle[_])
        max2, idx2 = Maximum(testY[_])
        if idx1 == idx2:
            counter += 1

    accuracy_angle = counter / len(pred_angle)
    print("angle:", accuracy_angle)
    with open('clf/clf_str.pickle', 'wb') as f1:
        pickle.dump(model_str, f1)
    with open('clf/clf_angle.pickle', 'wb') as f2:
        pickle.dump(model_angle, f2)
    print("ok")


def test():
    with open('clf/clf_str.pickle', 'rb') as f1:
        model_str = pickle.load(f1)
    with open('clf/clf_angle.pickle', 'rb') as f2:
        model_angle = pickle.load(f2)
        cap = cv2.VideoCapture(0)
        prevent_vibrate = 0
        while 1:
            ret, img = cap.read()
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # gray = cv2.resize(gray, (int(gray.shape[1]*resize), int(gray.shape[0]*resize)), interpolation=cv2.INTER_LINEAR)
                try:
                    landmark, shape, (x, y, w, h) = get_landmarks(gray)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # for _ in range(len(landmark)):
                    #     if _ < 28:
                    #         if _ % 4 == 0:
                    #             cv2.line(img, (int(landmark[_]+x), int(landmark[_ + 1]+y)), (int(landmark[_ + 4]+x), int(landmark[_ + 5]+y)), (0, 255, 0), 2)
                    strength = []
                    angle = []
                    for _ in range(len(landmark)):
                        if _ % 4 == 0:
                            print('x = %d' % landmark[_])
                        elif _ % 4 == 1:
                            print('y = %d' % landmark[_])
                        elif _ % 4 == 2:
                            print('strength = %d' % landmark[_])
                            strength.append(landmark[_])
                        else:
                            print('angle = %d' % landmark[_])
                            angle.append(landmark[_] / 180)
                    maxval, idx = Maximum(strength)
                    strength = np.array(strength / maxval, np.float32)
                    angle = np.array(angle, np.float32)
                    strength = np.reshape(strength, [1, -1])
                    angle = np.reshape(angle, [1, -1])
                    pred1 = model_str.predict(strength)
                    maxval1, index1 = Maximum(pred1[0])
                    pred2 = model_angle.predict(angle)
                    maxval2, index2 = Maximum(pred2[0])
                    if index1 == 0 and index2 == 0:
                        cv2.rectangle(img, (x, y - int(h * 0.1)), (x + int(w * 0.35), y), (0, 255, 0), -1)
                        cv2.putText(img, classifier[0], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                    2)
                        prevent_vibrate = 1
                    elif index1 == 1 and index2 == 1:
                        cv2.rectangle(img, (x, y - int(h * 0.1)), (x + int(w * 0.6), y), (0, 255, 0), -1)
                        cv2.putText(img, classifier[1], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                    2)
                        prevent_vibrate = 0
                    else:
                        if prevent_vibrate == 0:
                            cv2.rectangle(img, (x, y - int(h * 0.1)), (x + int(w * 0.6), y), (0, 255, 0), -1)
                            cv2.putText(img, classifier[1], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 255, 255),
                                        2)
                        else:
                            cv2.rectangle(img, (x, y - int(h * 0.1)), (x + int(w * 0.35), y), (0, 255, 0), -1)
                            cv2.putText(img, classifier[0], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 255, 255),
                                        2)
                    for (_x, _y) in shape:
                        cv2.circle(img, (_x, _y), 1, (0, 0, 255), -1)
                except:
                    print('no faces!!!')
                cv2.imshow('Face Recognition', img)
                cv2.waitKey(1)


if __name__ == "__main__":
    # train()
    test()
