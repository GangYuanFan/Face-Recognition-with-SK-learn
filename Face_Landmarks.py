import math
import dlib
from imutils import face_utils
import numpy as np
import cv2
fname = 'D:/DeepLearning/face/Face-LandMark_with_Dlib/facial-landmarks/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(fname)
l_eye_pct = 0.33
r_eye_pct = 0.66
eyes_level_pct = 0.4
def get_detect(gray):
    try:
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(gray, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return shape, (x, y, w, h)
    except:
        print("detect None")


def get_landmarks(gray):
    try:
        shape, (x, y, w, h) = get_detect(gray)
        xlist = []
        ylist = []
        frame = gray[y:y+h, x:x+w]
        for (_x, _y) in shape:
            xlist.append(_x)
            ylist.append(_y)
        left_eyeX = np.array(xlist[36:41])
        left_eyeY = np.array(ylist[36:41])
        right_eyeX = np.array(xlist[42:47])
        right_eyeY = np.array(ylist[42:47])
        left_center = (int(np.mean(left_eyeX)), int(np.mean(left_eyeY)))
        right_center = (int(np.mean(right_eyeX)), int(np.mean(right_eyeY)))
        # eye_center = (int((left_center[0] + right_center[0]) / 2), int((left_center[1] + right_center[1]) / 2))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        objpoints = np.array([(left_center[0] - x, left_center[1] - y),
                              (right_center[0] - x, right_center[1] - y),
                              (int(xmean) - x, int(ymean) - y)],
                             dtype=np.float32)
        imgpoints = np.array([(int(w * l_eye_pct), int(h * eyes_level_pct)),
                              (int(w * r_eye_pct), int(h * eyes_level_pct)),
                              (int(w / 2), int(h / 2))],
                             dtype=np.float32)
        M = cv2.getAffineTransform(objpoints, imgpoints)
        warped_image = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        objpoints = np.array([(int(warped_image.shape[1] * 0.1), int(warped_image.shape[0] * 0.33)),
                              (int(warped_image.shape[1] * 0.9), int(warped_image.shape[0] * 0.33)),
                              (int(warped_image.shape[1] * 0.9), int(warped_image.shape[0] * 0.66)),
                              (int(warped_image.shape[1] * 0.1), int(warped_image.shape[0] * 0.66))],
                             dtype=np.float32)
        imgpoints = np.array([(0, 0),
                              (int(warped_image.shape[1]), 0),
                              (int(warped_image.shape[1]), int(warped_image.shape[0])),
                              (0, int(warped_image.shape[0]))],
                             dtype=np.float32)
        M = cv2.getPerspectiveTransform(objpoints, imgpoints)
        warped_image = cv2.warpPerspective(warped_image, M, (warped_image.shape[1], warped_image.shape[0]))
        shape1, (x0, y0, w0, h0) = get_detect(warped_image)
        xlist = []
        ylist = []
        for (_x, _y) in shape1:  # 0~27 are face shape
            # cv2.circle(image, (_x, _y), 1, (0, 0, 255), -1)
            xlist.append(_x)
            ylist.append(_y)
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(_x - xmean) for _x in xlist]
        ycentral = [(_y - ymean) for _y in ylist]
        # cv2.circle(image, (int(xmean), int(ymean)), 2, (255, 255, 255), 0)

        landmarks_vectorised = []
        # landmarks_vectorised (x, y, length(point2central), angle)
        for _x, _y, _w, _z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(_w)
            landmarks_vectorised.append(_z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((_z, _w))
            dist = np.linalg.norm(coornp - meannp)  # find norm of vector
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(_y, _x) * 360) / (2 * math.pi))
        # for _ in range(len(landmarks_vectorised)):
        #   if _ % 4 == 0:
        #       cv2.line(image, (int(landmarks_vectorised[_]), int(landmarks_vectorised[_+1])), (int(xmean), int(ymean)), (0, 255, 0), 1)
        return landmarks_vectorised, shape, (x, y, w, h)
    except:
        print("no landmarks")