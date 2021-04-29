# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:59:36 2021

@author: lutzc
"""

#streamlit run linear_bayesian_regression.py
import base64
import matplotlib
from sympy import *
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import os

#init_printing()
    
def get_table_download_link_csv(df):
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="weights.csv" target="_blank">Download as csv file</a>'
    return href    

st.title('Bayesian Linear Regression')
st.write('If there is one algorithm that rules the world it is surely the least square regression. It is literally everywhere on :earth_americas:! Finance, science and technology use it extensively to predict, model and win the race of acute systems. The straight line fitting is commonly used by the average people but many more variants are living out there and are getting increasingly popular with the raise of Artificial Intelligence. In this application you will be able to experiment with linear least square which is both powerful and elegant. Under the bayesian framework the following can be naturally embraced collectively:')
st.write('* maximum likelihood (least square)')   
st.write('* maximum a posteriory (regularized least square)')
st.write('* posterior predictive')  
st.write('The data uncertainty is assumed to be gaussian, and equal everywhere (homoscedastic).')

st.header('Load data')
st.write('Firstly we need to set the stage of the problem. What we would like to predict? Based on what? Those are respectively called the target feature (y) and the input features (x,...).')
datasource = st.radio('',('From CSV', 'From Example'))

if datasource == 'From CSV' :
    st.write('Drop here the csv file containing the input and output features')
    file = st.file_uploader("Upload Files",type=['csv'])
    if file is not None:
        df = pd.read_csv(file, delimiter = ';')
    else:
        df = None
else:
    df= pd.DataFrame(
            {'x': np.array([1,2,3,4,5]),
             'y': np.array([2,5,10,17,26] + np.array([-0.008046  ,  0.00028632,  0.00710868,  0.00677746,  0.00552161])
             )})
#file = 'example.csv'

#

if df is not None:
    df
    x = df.iloc[:,-2].to_numpy()
    y = df.iloc[:,-1].to_numpy() 
    xlabel = df.columns[-2]
    ylabel = df.columns[-1]
    st.write("Let's have closer look :sleuth_or_spy:") 
    fig = px.scatter(x = x, y = y)
    fig.update_layout(xaxis_title= xlabel, yaxis_title= ylabel)
    fig
    st.header('Build additional features')
    st.write('Here the opportunity is given to build extra input features before rushing to the regression. Why would you to that? The reason is to increase the model complexity in order to more accurately predict the target feature. The functions to derive these features are called kernels and can be of all nature.')
    fun_str = st.text_input('List kernel functions (separated by commas) here - for example: x**2, sin(x), sqrt(x), 1 (for constant)', value='1')
    if fun_str != '':
        for fun in fun_str.split(','):
            df.insert(loc=0, column=fun, value=df.eval(fun))
    df
    
    
    X = df.iloc[:,0:-1].to_numpy()
    XTX = np.dot(X.T,X)
    N = X.shape[0]
    M = X.shape[1]
    eigvalues, _  = np.linalg.eig(XTX)
    
    st.header('Hyperparameters tuning')
    st.write('Bias or variance? That is the tradeoff. Hyperparameters are here to fine tune your models. Herein can be adjusted:')
    st.write('* Alpha which is the precision (or inverse variance) of the weights gaussian prior') 
    st.write('* Beta which is the precision (or inverse variance) of the data gaussian likelihood') 
    st.write('These two guys can be trained automatically by optimizing the data evidence in order for the model to be the most reasonanle in the light of the data. By the way the ratio alpha/beta is often called the regularization parameter, or lambda.')
    
    eps = np.finfo(float).eps
    alpha_trainable = st.checkbox('Make Alpha trainable',True)
    alpha = eps + st.slider(label = 'Alpha' + alpha_trainable * '_0', value = 1.0, min_value = 0.0, max_value = 100.0, step = 0.5)
    
    beta_trainable = st.checkbox('Make Beta trainable',True)
    beta = eps + st.slider(label = 'Beta' + beta_trainable * '_0', value = 1.0, min_value = 0.0, max_value = 100.0, step = 0.5)
    
    st.write('The variable gamma is derived from alpha, beta and the data only.')
    
    hpdf = pd.DataFrame({'alpha': [alpha], 'beta': [beta], 'gamma': [gamma]})
    
    thres = 1e-6
    for _ in range(1000):
        S0 = ( 1 / alpha ) * np.eye(M)
        SN = np.linalg.inv( np.linalg.inv(S0) + beta * XTX )
        mN = np.dot( SN, beta * np.dot(X.T, y) )
        prediction = np.dot(X,mN)
        residual = y - prediction
        lmbda = eigvalues / beta
        gamma = np.sum(lmbda / (lmbda + alpha))
        if alpha_trainable:
            alpha = gamma / np.dot(mN.T, mN)
        if beta_trainable:
            beta = 1.0 / ( np.sum(residual**2) / (N - gamma) )
        if np.abs(alpha - hpdf['alpha'].iloc[-1]) < thres and np.abs(beta - hpdf['beta'].iloc[-1]) < thres:
            break;
        hpdf = hpdf.append({'alpha': alpha, 'beta': beta, 'gamma': gamma},ignore_index=True)
        
    hpdf
    
    sigmaN = 1/beta + np.diag(np.dot(X,np.dot(SN,X.T)))
    muN = np.dot(X, mN )
    
    st.header('Results')
    st.write('By the power of bayesian regression, not only a maximum a posteriory estimate of the weights are available but also an estimate of their uncertainties. Consequently the prediction is also uncertain: part of the uncertainty originates from the model and part of uncertainty originates from the noise of the data.')
    
    red = 'rgba(255, 0, 0, 0.3)'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=muN - sigmaN,
        fill=None,
        mode='lines',
        line_color=red,
        ))
    fig.add_trace(go.Scatter(
        x = x,
        y = muN,
        fill ='tonexty', # fill area between trace0 and trace1
        mode ='markers + lines',
        line_color = 'red',
        fillcolor = red,
        ))
    fig.add_trace(go.Scatter(
        x = x,
        y = muN + sigmaN,
        fill ='tonexty', # fill area between trace0 and trace1
        mode ='lines',
        line_color = red,
        fillcolor = red,
        ))
    fig.add_trace(go.Scatter(
        x = x,
        y = y,
        mode ='markers',
        line_color = 'blue',
        ))
    fig.update_layout(showlegend=False)
    fig.update_layout(xaxis_title= xlabel, yaxis_title= ylabel)
    fig
    
    weights= pd.DataFrame({'Weights': df.columns[0:-1], 'Values': mN})
    weights
    
    st.markdown(get_table_download_link_csv(weights), unsafe_allow_html=True)
st.write('Based on Bishop, Christopher. (2006). Pattern Recognition and Machine Learning. 10.1117/1.2819119.')
st.write('Provided as is without any warranty.')
st.write('Created by Lutz Christophe - lutz.christophe@gmail.com - April 2021')