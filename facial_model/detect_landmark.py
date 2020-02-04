# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import numpy as np
import imutils
import time
import dlib
import cv2
from collections import OrderedDict
from pylab import *
from scipy.interpolate import interp1d
from skimage import color
from PIL import Image
from makeup import makeup

FACIAL_LANDMARKS_IDXS = OrderedDict([
	 #("mouth", (48, 68)),
	# ("right_eyebrow", (17, 22)),
	# ("left_eyebrow", (22, 27)),
	# ("right_eye", (36, 42)),
	# ("left_eye", (42, 48)),
	 #("nose", (27, 35)),
	("jaw", (0, 17))
])

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderProto = "./gender/deploy_gender.prototxt"
genderModel = "./gender/gender_net.caffemodel"
ageProto = "./age/deploy_age.prototxt"
ageModel = "./age/age_net.caffemodel"
genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_net = cv2.dnn.readNetFromCaffe(genderProto, genderModel)
age_net = cv2.dnn.readNetFromCaffe(ageProto, ageModel)
age, gender = None, None


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
print("[INFO] starting video stream...")
confidence_threshold = 0.5
vs = VideoStream(src=0).start()
time.sleep(2.0)
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    # print("running...")
    frame = vs.read()
    o_frame = frame.copy()
    blob = cv2.dnn.blobFromImage(o_frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    frame = imutils.resize(frame, width=500)
    m = makeup(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) > 0:
        gender_net.setInput(blob)
        age_net.setInput(blob)
        gender_preds = gender_net.forward()
        age_preds = age_net.forward()
        if gender_preds[0].max() > 0.900:
            gender = genderList[gender_preds[0].argmax()]

        if age_preds[0].max() > 0.900:
            age = ageList[age_preds[0].argmax()]
    if age and gender:
        label = "{},{}".format(gender, age)
    # print(label)

    upper_lip_landmarks = [49, 50, 51, 52, 53, 54, 55, 61, 62, 63, 64, 65]
    lower_lip_landmarks = [49, 55, 56, 57, 58, 59, 60, 61, 65, 66, 67, 68]
    # (h, w) = frame.shape[:2]
    for (i, rect) in enumerate(rects):

        # print(i)
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        # print(rect)
        shape_ = predictor(gray, rect)
        # print(shape)
        # k = face_utils.FACIAL_LANDMARKS_IDXS(shape)
        # print(k)
        shape = face_utils.shape_to_np(shape_)

        landmark = np.empty([68, 2], dtype=int)
        for i in range(68):
            landmark[i][0] = shape_.part(i).x
            landmark[i][1] = shape_.part(i).y

        frame = m.apply_makeup(landmark)

    # show the output frame
    if age and gender:
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# show the output image with the face detections + facial landmarks
# cv2.imshow("Output", image)
# cv2.waitKey(0)

