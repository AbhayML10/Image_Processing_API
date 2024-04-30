from fastapi import FastAPI, File, UploadFile, Request, Response, Depends, HTTPException, Cookie
from fastapi import Form
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from Process.functions import predict_from_model
# from jproperties import Properties

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], 
    allow_credentials = True,
    allow_methods = ["GET", "POST", "PUT", "DELETE"], 
    allow_headers = ["*"],  
)



@app.post('/Pred_Img')
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
    res = predict_from_model(img)
    return {"Result":res}




if __name__ == '__main__':
    uvicorn.run('__main__:app',host="127.0.0.1",port = 8000)