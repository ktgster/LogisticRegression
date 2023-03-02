# import libraries
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi import Request
import json
import cgi

app = FastAPI()
classifier = joblib.load('model.pkl')

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    with open('index.html') as f:
        html = f.read()
    return html

class CancerData(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

@app.post('/predict')
async def predict_cancer(request: Request):
    form = await request.form()
    data = {}
    for field in form:
        data[field] = form[field]

    data = json.loads(json.dumps(data))

    data = CancerData(**data)
    radius_mean = data.radius_mean
    texture_mean = data.texture_mean
    perimeter_mean = data.perimeter_mean
    area_mean = data.area_mean
    smoothness_mean = data.smoothness_mean
    compactness_mean = data.compactness_mean
    concavity_mean = data.concavity_mean
    concave_points_mean = data.concave_points_mean
    symmetry_mean = data.symmetry_mean
    fractal_dimension_mean = data.fractal_dimension_mean
    radius_se = data.radius_se
    texture_se = data.texture_se
    perimeter_se = data.perimeter_se
    area_se = data.area_se
    smoothness_se = data.smoothness_se
    compactness_se = data.compactness_se
    concavity_se = data.concavity_se
    concave_points_se = data.concave_points_se
    symmetry_se = data.symmetry_se
    fractal_dimension_se = data.fractal_dimension_se
    radius_worst = data.radius_worst
    texture_worst = data.texture_worst
    perimeter_worst = data.perimeter_worst
    area_worst = data.area_worst
    smoothness_worst = data.smoothness_worst
    compactness_worst = data.compactness_worst
    concavity_worst = data.concavity_worst
    concave_points_worst = data.concave_points_worst
    symmetry_worst = data.symmetry_worst
    fractal_dimension_worst = data.fractal_dimension_worst
    
    prediction = classifier.predict([[radius_mean, texture_mean, perimeter_mean, area_mean,
    smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,
    symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se,
    area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, 
    symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst,
    area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
    symmetry_worst, fractal_dimension_worst]])
                                    
    prediction_value = prediction[0]
    if prediction_value >= 0.5:
        prediction_label = 'Malignant'
    else:
        prediction_label = 'Benign'

    return {'prediction': prediction_label}

        

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


