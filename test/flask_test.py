from flask import Flask, request
app = Flask(__name__)

import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt
#import argparse
#import imutils
import time
import json


def mesurement_estimation(img):
    DATASET = 'COCO' # Specify what kind of model was trained. It could be (COCO, MPI) depends on dataset.

    THR = 0.1 # Threshold value for pose parts heat map
    WIDTH = 200 # Resize input to specific width (default: 368).
    HEIGHT = 200 # Resize input to specific height (default: 368).

    if DATASET == 'COCO':
        PROTO = './pose/coco/deploy_coco.prototxt'
        MODEL = './pose/coco/pose_iter_440000.caffemodel'
        BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                       "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                       "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                       "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    #    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    #                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    #                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    #                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    #                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
        POSE_PAIRS = [ ["LShoulder", "RShoulder"], ["LShoulder", "RHip"], ["RShoulder", "LHip"],
                       ["LHip", "RHip"], ["LShoulder", "LHip"], ["RShoulder", "RHip"] ]

    elif DATASET == 'MPI':
        PROTO = './pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt'
        MODEL = './pose/mpi/pose_iter_160000.caffemodel'
        BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                       "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                       "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                       "Background": 15 }

        POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                       ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                       ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                       ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    else:
        PROTO = './pose/body_25/body_25_deploy.prototxt'
        MODEL = './pose/body_25/pose_iter_584000.caffemodel'
        BODY_PARTS ={"Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,"LShoulder":5,"LElbow":6,"LWrist":7,"MidHip":8,"RHip":9,"RKnee":10,"RAnkle":11,"LHip":12,"LKnee":13,"LAnkle":14,"REye":15,"LEye":16,"REar":17,"LEar":18,"LBigToe":19,"LSmallToe":20,"LHeel":21,"RBigToe":22,"RSmallToe":23,"RHeel":24,"Background":25}

        POSE_PAIRS =[ ["Neck","MidHip"],   ["Neck","RShoulder"],   ["Neck","LShoulder"],   ["RShoulder","RElbow"],   ["RElbow","RWrist"],   ["LShoulder","LElbow"],   ["LElbow","LWrist"],   ["MidHip","RHip"],   ["RHip","RKnee"],  ["RKnee","RAnkle"], ["MidHip","LHip"],  ["LHip","LKnee"], ["LKnee","LAnkle"],  ["Neck","Nose"],   ["Nose","REye"], ["REye","REar"],  ["Nose","LEye"], ["LEye","LEar"],   
    ["RShoulder","REar"],  ["LShoulder","LEar"],   ["LAnkle","LBigToe"],["LBigToe","LSmallToe"],["LAnkle","LHeel"], ["RAnkle","RBigToe"],["RBigToe","RSmallToe"],["RAnkle","RHeel"] ]

    inWidth = WIDTH
    inHeight = HEIGHT

    net = cv.dnn.readNetFromCaffe(PROTO, MODEL)

    frame = cv.imdecode(np.frombuffer(img.read(), dtype='uint8'), -1)
    frame = frame[:,:,:3]
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    start_t = time.time()
    out = net.forward()

    #print("time is ",time.time()-start_t)
    # print(inp.shape)
    #kwinName="Pose Estimation Demo: Cv-Tricks.com"
    #cv.namedWindow(kwinName, cv.WINDOW_AUTOSIZE)
    #assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > THR else None)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (255, 74, 0), 3)
            cv.ellipse(frame, points[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.putText(frame, str(idFrom), points[idFrom], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)
            cv.putText(frame, str(idTo), points[idTo], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)

    shoulders = abs(points[BODY_PARTS["LShoulder"]][0] - points[BODY_PARTS["RShoulder"]][0])
    hips = abs(points[BODY_PARTS["LHip"]][0] - points[BODY_PARTS["RHip"]][0])

    #print('Shoulders: '+ str(shoulders), 'Hips: ' + str(hips))

    shouldersL = np.array((points[BODY_PARTS["LShoulder"]][0] ,points[BODY_PARTS["LShoulder"]][1]))
    shouldersR = np.array((points[BODY_PARTS["RShoulder"]][0] ,points[BODY_PARTS["RShoulder"]][1]))

    shouldersEucl = np.linalg.norm(shouldersL - shouldersR)

    hipsL = np.array((points[BODY_PARTS["LHip"]][0] ,points[BODY_PARTS["LHip"]][1]))
    hipsR = np.array((points[BODY_PARTS["RHip"]][0] ,points[BODY_PARTS["RHip"]][1]))

    hipsEucl = np.linalg.norm(hipsL - hipsR)
    result = {
        'shoulders': shouldersEucl,
        'hips': hipsEucl
        
    }
    result_json = json.dumps(result)
   
    
   # return 'Shoulders Eucl: '+ str(shouldersEucl) + 'Hips Eucl: ' + str(hipsEucl)
    return(result_json)

@app.route('/', methods=['POST'])
def hello_world():
    if 'file' not in request.files:
        return "No file"
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    #return repr(file.stream.read())
    return mesurement_estimation(file.stream)
