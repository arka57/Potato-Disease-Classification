from flask import Flask,render_template,request


import numpy as np
from utils import transform_images,predict



app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_disease():

    f=request.files['image']
    fn=f.filename
    image_bytes=f.read()
    image=transform_images(image_bytes)
    result=predict(image)
    return render_template('index.html',result=result)    
    

if __name__=='__main__':
    app.run(debug=True)