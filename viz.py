from config import config
import random
import os
import cv2
import matplotlib.pyplot as plt

def viz_display(path, list_bboxes):
    path = path.strip()
    image_id = "%s/%s"%(path.split('/')[-2], path.split('/')[-1])
    img = cv2.imread(path)
    bboxes = list_bboxes[image_id]
    #if img == None:
        #print('Img not loaded currently!')
        #return None

    for obj in bboxes:
        x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x1']+obj['w'], obj['y1']+obj['h']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)

    cv2.imshow('visualize img', img)
    cv2.waitKey(-1)

def viz_landmarks(path, list_landmarks, show_bounding=True):
    path = path.strip()
    image_id = "%s/%s"%(path.split('/')[-2], path.split('/')[-1])
    img = cv2.imread(path)
    print(img.shape)
    
    for obj in list_landmarks[image_id]:
        landmarks = obj['landmarks']
        for c in landmarks:
            cv2.circle(img, (int(c[0]), int(c[1])), 2, (0,0,255), -1)

    if show_bounding:
        for obj in list_landmarks[image_id]:
            bbox = obj['bbox']
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0,0,255), 2)   

    cv2.imshow('visualization of landmarks',img)
    cv2.waitKey(-1)
