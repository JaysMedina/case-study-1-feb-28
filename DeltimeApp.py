# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:16:29 2022

@author: jmmedina
"""
#%% Run Part 1 app
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import dash_html_components as html

app = Flask(__name__)

# define deltime csv to df to json
df_init = pd.read_csv ('deliverytime.csv')
deltime = df_init.to_json(orient="records")

colors = {'background':'#111111', 'text': '#7FDBFF'}

stylediv={'textAlign': 'center', 'color': colors['text']}

app.layout = html.Div(children=[
    html.H1(children='Hello Dash',style=stylediv),
    html.Div(children='Dash: A web application framework for Python',style=stylediv),
   ], style={'backgroundColor':colors['background']}
)

# find next id (used for appending)
def _find_next_id():
    return max(entry["id"] for entry in deltime) + 1

# compute for linear regression model, takes json 
def _make_lm(jsondata):
    df = pd.read_json(jsondata,orient='columns')
    X = df[["ncases","distance"]]
    y = df[["deltime"]]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    #Learn the coefficients
    regressor=LinearRegression()
    regressor.fit(X_train,y_train)
    b0 = regressor.intercept_[0]
    b1 = regressor.coef_[0][0]
    b2 = regressor.coef_[0][1]
    coefflist = [b0, b1, b2]
    return coefflist

# show current dataset
@app.get("/deltime")
def get_data():
    headers = request.headers
    auth = headers.get("api-key")
    if auth == 'mypassword':
        return jsonify(deltime)
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401

# predict using linear regression model
@app.get('/predict/<ncases>/<distance>')
def predict_deltime(ncases,distance):
    ncases = int(ncases)
    distance = int(distance)
    headers = request.headers
    auth = headers.get("api-key")
    if auth == 'mypassword':
        model = _make_lm(deltime)
        prediction = model[0] + model[1]*(ncases) + model[2]*(distance)
        return str(prediction)
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401
    
# post function adds to json to df to csv
@app.post('/deltime')
def add_data():
    headers = request.headers
    auth = headers.get("api-key")
    if auth == 'mypassword':
        if request.is_json:
            global deltime
            entry = request.get_json()
            entry = str(entry)
            deltime = deltime + entry
            return entry, 201
        return {"error":"Request must be JSON"}, 415
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401
    
if __name__ == '__main__':
    app.run()
