import streamlit as st
import cv2 as cv
import argparse
import numpy as np
import sys
import time
from threading import Thread
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

from common import *
from tf_text_graph_common import readTextMessage
from tf_text_graph_ssd import createSSDGraph
from tf_text_graph_faster_rcnn import createFasterRCNNGraph

backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV,
            cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)
targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD, cv.dnn.DNN_TARGET_HDDL,
           cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

mean = [104, 177, 123]
rgb = False
asyncN = 0
scale = 1.0

# If config specified, try to load it as TensorFlow Object Detection API's pipeline.
config = readTextMessage('opencv_face_detector.prototxt')

# Load a network
net = cv.dnn.readNet('res10_300x300_ssd_iter_140000.caffemodel', 'opencv_face_detector.prototxt', None)
backend = 0
target = 0
net.setPreferableBackend(backend)
net.setPreferableTarget(target)
outNames = net.getUnconnectedOutLayersNames()

confThreshold = 0.5
nmsThreshold = 0.4
st.subheader('Phát hiện khuôn mặt DNN OpenCV Streamlit')
FRAME_WINDOW = st.image([])

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf


        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    if lastLayer.type == 'DetectionOutput':
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        for out in outs:
            for detection in out[0, 0]:
                confidence = detection[2]
                if confidence > confThreshold:
                    left = int(detection[3])
                    top = int(detection[4])
                    right = int(detection[5])
                    bottom = int(detection[6])
                    width = right - left + 1
                    height = bottom - top + 1
                    if width <= 2 or height <= 2:
                        left = int(detection[3] * frameWidth)
                        top = int(detection[4] * frameHeight)
                        right = int(detection[5] * frameWidth)
                        bottom = int(detection[6] * frameHeight)
                        width = right - left + 1
                        height = bottom - top + 1
                    classIds.append(int(detection[1]) - 1)  # Skip background label
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    elif lastLayer.type == 'Region':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    # NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    # or NMS is required if number of outputs > 1
    if len(outNames) > 1 or lastLayer.type == 'Region' and backend != cv.dnn.DNN_BACKEND_OPENCV:
        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box  = boxes[class_indices].tolist()
            nms_indices = cv.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            nms_indices = nms_indices[:, 0] if len(nms_indices) else []
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

# Process inputs
winName = 'Deep learning object detection in OpenCV'
#cv.namedWindow(winName, cv.WINDOW_NORMAL)

def callback(pos):
    global confThreshold
    confThreshold = pos / 100.0

cap = cv.VideoCapture(0)

class QueueFPS(queue.Queue):
    def __init__(self):
        queue.Queue.__init__(self)
        self.startTime = 0
        self.counter = 0

    def put(self, v):
        queue.Queue.put(self, v)
        self.counter += 1
        if self.counter == 1:
            self.startTime = time.time()

    def getFPS(self):
        return self.counter / (time.time() - self.startTime)


process = True

#
# Frames capturing thread
#
framesQueue = QueueFPS()
def framesThreadBody():
    global framesQueue, process

    while process:
        hasFrame, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        if not hasFrame:
            break
        framesQueue.put(frame)


#
# Frames processing thread
#
processedFramesQueue = queue.Queue()
predictionsQueue = QueueFPS()
def processingThreadBody():
    global processedFramesQueue, predictionsQueue, process

    futureOutputs = []
    while process:
        # Get a next frame
        frame = None
        try:
            frame = framesQueue.get_nowait()

            if asyncN:
                if len(futureOutputs) == asyncN:
                    frame = None  # Skip the frame
            else:
                framesQueue.queue.clear()  # Skip the rest of frames
        except queue.Empty:
            pass


        if not frame is None:
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]

            # Create a 4D blob from a frame.
            inpWidth = 300 
            inpHeight = 300
            blob = cv.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=rgb, ddepth=cv.CV_8U)
            processedFramesQueue.put(frame)

            # Run a model
            net.setInput(blob, scalefactor=scale, mean=mean)
            if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                frame = cv.resize(frame, (inpWidth, inpHeight))
                net.setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')

            if asyncN:
                futureOutputs.append(net.forwardAsync())
            else:
                outs = net.forward(outNames)
                predictionsQueue.put(np.copy(outs))

        while futureOutputs and futureOutputs[0].wait_for(0):
            out = futureOutputs[0].get()
            predictionsQueue.put(np.copy([out]))

            del futureOutputs[0]


framesThread = Thread(target=framesThreadBody)
framesThread.start()

processingThread = Thread(target=processingThreadBody)
processingThread.start()

#
# Postprocessing and rendering loop
#
while True:
    try:
        # Request prediction first because they put after frames
        outs = predictionsQueue.get_nowait()
        frame = processedFramesQueue.get_nowait()

        postprocess(frame, outs)

        # Put efficiency information.
        if predictionsQueue.counter > 1:
            label = 'Camera: %.2f FPS' % (framesQueue.getFPS())
            cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            label = 'Network: %.2f FPS' % (predictionsQueue.getFPS())
            cv.putText(frame, label, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            label = 'Skipped frames: %d' % (framesQueue.counter - predictionsQueue.counter)
            cv.putText(frame, label, (0, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        FRAME_WINDOW.image(frame)
    except queue.Empty:
        pass


process = False
framesThread.join()
processingThread.join()
