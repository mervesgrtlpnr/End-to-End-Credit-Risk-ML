from fastapi import FastAPI,Request,Response
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel


app = FastAPI()

#templates
templates = Jinja2Templates(directory="templates")

with open("credit_risk_complete.pkl","rb") as f :
    save_data = pickle.load(f)
    model = save_data['model']

class Features(BaseModel):
    person_age: int
    loan_intent:str
    loan_grade:str
    person_home_ownership:str
    person_income: int
    person_emp_length:float
    loan_amnt: int
    loan_int_rate:float
    loan_percent_income:float
    cb_person_default_on_file:str
    cb_person_cred_hist_length:int


@app.get("/",response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/predict")
async def predict(features: Features) :
    input_data = pd.DataFrame([features.model_dump()])
    print(input_data)

    input_data['cb_person_default_on_file'] = input_data['cb_person_default_on_file'].map({'N':0 , 'Y':1})

    prediction = model.predict(input_data)
    return {'prediction': int(prediction[0])}