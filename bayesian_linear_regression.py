# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:59:36 2021

@author: lutzc
"""

#streamlit run bayesian_linear_regression.py
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



def get_table_download_link_csv(df, text = 'download'):
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="weights.csv" target="_blank">'+ text +'</a>'
    return href    



st.title('Bayesian Linear Regression')

st.write('If there is one algorithm that has experienced massive adoption across all domains, it is surely the least-square regression. It is everywhere on :earth_americas:! Finance, science, and technology use it extensively to predict, model, and win the race of high-accuracy systems. The straight-line fitting is a typical entry point of the field but many more variants are living out there and are getting increasingly popular with the rise of Artificial Intelligence.')

st.write('In this application, you will be able to experiment with linear models which are both powerful and elegant. A linear model $\mathcal{M}$  attempts to explain the target variable $y$ from a weighted linear combination of the input features $(x_j)_{1..M}$ with the weights $(w_j)_{1..M}$. The general form of a linear model is as follow:')

st.write(r'$\mathcal{M} : y = \sum{w_jx_j} + \epsilon$ where $\epsilon$ is a 0-mean Gaussian noise of precision $\beta=\frac{1}{\sigma^2}$.')

st.write('The goal in the following is to determine the values of weights.')



st.header('Load data')

# choose file
st.write('Import your data ($y$ shall be the last column and $x$ the next to last) or use a predefined data set.')
datasource = st.radio('Source',('From CSV', 'From Example: Parabola', 'From Example: Sine'), index=2)

if datasource == 'From CSV' :
    file = st.file_uploader("Upload Files",type=['csv'])
    kernels = '1'
elif datasource == 'From Example: Parabola' :
    file = 'samples/parabola.csv'
    kernels = '1, x**2'
elif datasource == 'From Example: Sine' :
    file = 'samples/sine.csv'
    kernels = '1, x**2, x**3'
else:
    file = None

# create dataframe
if file is not None:
    df = pd.read_csv(file)
else:
    df = None


# process dataframe
if df is not None:
    df
    x = df.iloc[:,-2].to_numpy()
    y = df.iloc[:,-1].to_numpy() 
    xlabel = df.columns[-2]
    ylabel = df.columns[-1]
    st.write("Let's have a closer look :sleuth_or_spy:") 
    fig = px.scatter(x = x, y = y)
    fig.update_layout(xaxis_title= xlabel, yaxis_title= ylabel)
    fig
    
    
    st.header('Build additional features')
    st.write('This section offers the opportunity to build extra input features with custom kernels functions applied on the loaded data in order to boost the model representational capability.')
    fun_str = st.text_input('List of kernel functions (separated by commas) - for example: x**2, sin(x), sqrt(x), 1', value = kernels)
    if fun_str != '':
        for fun in fun_str.split(','):
            df.insert(loc=0, column=fun.strip(), value=df.eval(fun))
    df
    
    
    
    X = df.iloc[:,0:-1].to_numpy()
    XTX = np.dot(X.T,X)
    N = X.shape[0]
    M = X.shape[1]
    eigvalues, _  = np.linalg.eig(XTX)
    
    
    
    
    st.header('Hyperparameters tuning')
    st.write('Bias or variance? That is the tradeoff. Hyperparameters are here to fine-tune your model. Herein can be adjusted:')
    st.write(r'- $\alpha$ which is the precision (or inverse variance)  of the weights prior.') 
    st.write(r'- $\beta$ which is the precision (or inverse variance) of the targets.') 
    st.write(r'Note that the ratio $\alpha/\beta$ is called  regularization parameter, or $\lambda$.')
    
    
    eps = np.finfo(float).eps
    alpha_trainable = st.checkbox('Make Alpha trainable',True)
    alpha = eps + st.slider(label = 'Alpha' + alpha_trainable * '_0', value = 1.0, min_value = 0.0, max_value = 100.0, step = 0.5)
    
    beta_trainable = st.checkbox('Make Beta trainable',True)
    beta = eps + st.slider(label = 'Beta' + beta_trainable * '_0', value = 1.0, min_value = 0.0, max_value = 100.0, step = 0.5)
    
    hpdf = pd.DataFrame({'alpha': [alpha], 'beta': [beta], 'gamma': [gamma], 'lambda': [alpha/beta]})
    
    thres = 1e-6
    overflow = 1e12
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
        lmbda = alpha/beta
        if (np.abs(alpha - hpdf['alpha'].iloc[-1]) < thres and np.abs(beta - hpdf['beta'].iloc[-1]) < thres) or alpha > overflow or beta > overflow:
            break;
        hpdf = hpdf.append({'alpha': alpha, 'beta': beta, 'gamma': gamma, 'lambda': alpha/beta},ignore_index=True)     
    hpdf
    
    sigmaN = np.sqrt(1/beta + np.diag(np.dot(X,np.dot(SN,X.T))))
    muN = np.dot(X, mN )
    
    st.header('Results')
    st.write('The plot below shows the prediction with the maximum a posteriori weights, and the posterior predictive distribution which indicates the (one standard deviation) uncertainty related to both the model and the noise of the target values.')
    
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
    
    st.write(r'$y \approx$')
    for j, label in enumerate(weights['Weights']):
        st.write(weights['Values'][j], ' $' + label + (j!=M-1)*'+' + '$')
        
        
    st.write('')
    st.markdown(get_table_download_link_csv(weights, 'Download weights as CSV.'), unsafe_allow_html=True)




st.write('')
st.write('')
st.write('')

st.write('---')
if st.button("Learn More about Bayesian Regression"):
    st.write('Bayesian regression is a very powerful framework at the origin of least-square and regularized least square fitting methods. The hearth of the reasoning is the Bayes formula:')

    st.write(r'$P(W|Y) = \frac{P(Y|W)P(W)}{P(Y)}$')

    st.write('Conditionning every probabilities above on $X$ and $\mathcal{M}$ yields the main formula of interest:')

    st.write(r'$P(W|Y,X,\mathcal{M}) = \frac{P(Y|W,X,\mathcal{M})P(W|X,\mathcal{M})}{P(Y|X,\mathcal{M})}$')

    st.write('where $Y = [y_i]$, $X = [x_{i,j}]$ and $W = [w_j]$ are respectively the targets, the features and the weights matrices')
    
    st.write('Let us analyze each terms:')
    st.write(r'- $P(W|X,\mathcal{M})$ is the weight prior, which is taken as a 0-mean Gaussian distribution of precision parameter $\alpha$ herein (it does not depend on $X$).')
    st.write(r'- $P(Y|W,X,\mathcal{M})$ is the likelihood, which in the light of the model $\mathcal{M}$ is a $XW$-mean diagonal Gaussian of precision $\beta$.')
    st.write(r'- $P(Y|X,\mathcal{M})$ is the model evidence which indicates how likely the model was to generate the target samples, when marginalizing the weights.')
    st.write(r'- $P(W|Y,X,\mathcal{M})$ is the weights posterior probability distribution which is what we want to infer.')
    st.write(r'Maximizing (w.r.t. $W$) the probability $P(W|Y,X,\mathcal{M})$ yields the MAP weights. Marginalizing (w.r.t. $W$) the model prediction from $\mathcal{M}$ yields the measure of the uncertainty in the prediction. ')
#    st.write(r'Maximizing (w.r.t. $W$) $P(Y|X,\mathcal{M})$  weights combination (actually a full continuous set) describing every possible models of the form of $\mathcal{M}$.')
    st.write(r'The $\alpha$ and $\beta$ hyperparameter can be set automatically by maximizing the model evidence. Refer to the following book for more details: ')
    st.write(':green_book: Bishop, Christopher. (2006). Pattern Recognition and Machine Learning. 10.1117/1.2819119.')
    
    
    
st.write('---')



st.write('Provided as is without any warranty.')
st.write('Created by Lutz Christophe - lutz.christophe@gmail.com - April 2021')



