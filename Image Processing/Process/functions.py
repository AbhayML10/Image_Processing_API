import tensorflow as tf
import cv2
import numpy as np
import os
# from tensorflow.keras.models import load_model
import torch
from ultralytics import YOLO



def predict_from_model(img):
    img = cv2.resize(img, (640,640))
    print(np.array(img).shape)
    img = np.array(img)
    # img = np.expand_dims(img,axis=2)
    # img = np.expand_dims(img, axis = 0)
    print(img.shape)
    # model_path = os.path.join('..','Models','Abnormality_Best.keras')
    model = YOLO("Models/last_last.pt")
    # model = load_model(model_path)
    y_pred = model.predict(img,show=False,conf=0.6,visualize=True,save=False,show_labels=True,show_conf=True,line_width=1)
    mask = get_mask(y_pred)
    bbx = get_bound_box(mask)
    crop_images = get_cropped_imgs(bbx,img)
    res = y_pred[0].plot()
    return {"Predicted_Image":res,"Cropped_Images":crop_images}


def get_mask(res):
    masks = res[0].masks.data
    box = res[0].boxes.data
    clss = box[:,5]
    print(clss)
    ind = torch.where(clss!=-1)
    mas = masks[ind]
    ob_mas = torch.any(mas,dim=0).int()*255
    return ob_mas

def get_bound_box(mask):
    cv2.imwrite("img_1.jpg",mask.cpu().numpy())
    mask_im = cv2.imread("img_1.jpg")
    mask_gray = cv2.cvtColor(mask_im,cv2.COLOR_BGR2GRAY)
    _, thresh_h = cv2.threshold(mask_gray, 215,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbx=[]
    for con in contours:
        x,y,w,h = cv2.boundingRect(con)
        print(x,y,w,h)
        bbx.append([x,y,w,h])
    return bbx


def get_cropped_imgs(bbx,img):
    crop_imgs = []
    for i in bbx:
        crop_imgs.append(img[i[1]-15:i[1]+i[3]+10, i[0]-15:i[0]+i[2]+10])
    return crop_imgs