# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:59:36 2021
@author: lutzc
"""

# streamlit run bayesian_linear_regression.py
import base64
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math

def get_table_download_link_csv(df, text='download'):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="weights.csv" target="_blank">{text}</a>'
    return href    

st.set_page_config(page_title="Bayesian Linear Regression", layout="wide")
st.title('Bayesian Linear Regression')

st.write('If there is one algorithm that has experienced massive adoption across all domains, it is surely the least-square regression. It is everywhere on :earth_americas:! Finance, science, and technology use it extensively to predict, model, and win the race of high-accuracy systems. The straight-line fitting is a typical entry point of the field but many more variants are living out there and are getting increasingly popular with the rise of Artificial Intelligence.')

st.write('In this application, you will be able to experiment with linear models which are both powerful and elegant. A linear model $\mathcal{M}$  attempts to explain the target variable $y$ from a weighted linear combination of the input features $(x_j)_{1..M}$ with the weights $(w_j)_{1..M}$. The general form of a linear model is as follow:')

st.latex(r'\mathcal{M} : y = \sum w_j x_j + \epsilon \quad \text{where } \epsilon \text{ is a 0-mean Gaussian noise of precision } \beta=\frac{1}{\sigma^2}.')

st.write('The goal in the following is to determine the values of weights.')

st.header('Load data')

st.write('Import your data ($y$ shall be the last column and $x$ the next to last) or use a predefined data set.')
datasource = st.radio('Source', ('From CSV', 'From Example: Parabola', 'From Example: Sine'), index=2)

df0 = None
if datasource == 'From CSV':
    file = st.file_uploader("Upload Files", type=['csv'])
    if file is not None:
        df0 = pd.read_csv(file)
elif datasource == 'From Example: Parabola':
    x_gen = np.linspace(-5, 5, 150)
    y_gen = 2.5 * x_gen**2 - 1.2 * x_gen + 0.5 + np.random.normal(0, 5, 150)
    df0 = pd.DataFrame({'x': x_gen, 'y': y_gen})
elif datasource == 'From Example: Sine':
    x_gen = np.linspace(0, 1, 150)
    y_gen = np.sin(2 * np.pi * x_gen) + np.random.normal(0, 0.2, 150)
    df0 = pd.DataFrame({'x': x_gen, 'y': y_gen})

if df0 is not None:
    st.dataframe(df0.head())
    x = df0.iloc[:, -2].to_numpy()
    y = df0.iloc[:, -1].to_numpy() 
    xlabel = df0.columns[-2]
    ylabel = df0.columns[-1]
    
    st.write("Let's have a closer look 🕵️")
    fig = px.scatter(x=x, y=y)
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
    st.plotly_chart(fig, use_container_width=True)
    
    st.header('Select and build features')
    st.write('This section offers the opportunity to build extra input features with custom kernels functions applied on the loaded data in order to boost the model representational capability.')
    
    df = pd.DataFrame({})
    df.insert(loc=0, column=ylabel, value=y)
    
    if datasource == 'From CSV':
        default_val = ', '.join(df0.columns[:-1]) + ', 1'
    elif datasource == 'From Example: Parabola':
        default_val = '1, x, x**2'
    elif datasource == 'From Example: Sine':
        default_val = '1, x, x**2, x**3'
        
    fun_str = st.text_input('List of kernel functions (separated by commas) - for example: x**2, sin(x), sqrt(x), 1 for a constant.', value=default_val)
    
    # Setup safe evaluation namespace
    namespace = {col: df0[col].values for col in df0.columns}
    namespace['x'] = x
    namespace['y'] = y
    for k in dir(np):
        if not k.startswith('_'): namespace[k] = getattr(np, k)
    for k in dir(math):
        if not k.startswith('_'): namespace[k] = getattr(math, k)
        
    if fun_str != '':
        for fun in fun_str.split(','):
            col_name = fun.strip()
            if not col_name: continue
            try:
                val = eval(col_name, {"__builtins__": None}, namespace)
                df.insert(loc=0, column=col_name, value=val)
            except Exception as e:
                st.error(f"Error evaluating '{col_name}': {e}")
    
    st.dataframe(df.head())
    
    X = df.iloc[:, 0:-1].to_numpy()
    N = X.shape[0]
    M = X.shape[1]
    
    # Precompute XTX and eigenvalues
    XTX = np.dot(X.T, X)
    eigvalues = np.linalg.eigvalsh(XTX) # More stable for symmetric matrices
    
    st.header('Hyperparameters tuning')
    st.write('Bias or variance? That is the tradeoff. Hyperparameters are here to fine-tune your model. Herein can be adjusted:')
    st.write(r'- $\alpha$ which is the precision (or inverse variance)  of the weights prior.') 
    st.write(r'- $\beta$ which is the precision (or inverse variance) of the targets.') 
    st.write(r'Note that the ratio $\alpha/\beta$ is called  regularization parameter, or $\lambda$.')
    st.write(r'Note that $\gamma$ is the effective number of parameters.')

    eps = np.finfo(float).eps
    alpha_trainable = st.checkbox('Make Alpha trainable', True)
    alpha = eps + st.slider(label='Alpha' + ('_0' if alpha_trainable else ''), value=1.0, min_value=0.0, max_value=100.0, step=0.5)
    
    beta_trainable = st.checkbox('Make Beta trainable', True)
    beta = eps + st.slider(label='Beta' + ('_0' if beta_trainable else ''), value=1.0, min_value=0.0, max_value=100.0, step=0.5)
    
    hpdf_list = [{'alpha': alpha, 'beta': beta, 'gamma': 0.0, 'lambda': alpha/beta}]
    
    thres = 1e-6
    overflow = 1e12
    
    for _ in range(1000):
        S0_inv = alpha * np.eye(M)
        SN_inv = S0_inv + beta * XTX
        SN = np.linalg.inv(SN_inv)
        mN = np.dot(SN, beta * np.dot(X.T, y))
        prediction = np.dot(X, mN)
        residual = y - prediction
        
        # Bishop Eq 3.87 & 3.91: eigenvalues of beta * X^T X
        lmbda = eigvalues * beta 
        gamma = np.sum(lmbda / (lmbda + alpha))
        
        alpha_new = alpha
        beta_new = beta
        
        if alpha_trainable:
            alpha_new = gamma / (np.dot(mN.T, mN) + eps)
        if beta_trainable:
            # Safeguard against N - gamma <= 0
            beta_new = max(N - gamma, 1e-4) / (np.sum(residual**2) + eps)
            
        if (np.abs(alpha_new - alpha) < thres and np.abs(beta_new - beta) < thres) or alpha_new > overflow or beta_new > overflow:
            alpha = alpha_new
            beta = beta_new
            hpdf_list.append({'alpha': alpha, 'beta': beta, 'gamma': gamma, 'lambda': alpha/beta})
            break
            
        alpha = alpha_new
        beta = beta_new
        hpdf_list.append({'alpha': alpha, 'beta': beta, 'gamma': gamma, 'lambda': alpha/beta})
        
    hpdf = pd.DataFrame(hpdf_list)
    st.dataframe(hpdf.tail())
    
    # Efficient O(N M^2) variance calculation instead of O(N^2 M)
    sigmaN = np.sqrt(1/beta + np.sum((X @ SN) * X, axis=1))
    muN = np.dot(X, mN)
    
    st.header('Results')
    st.write('The plot below shows the prediction with the maximum a posteriori weights, and the posterior predictive distribution which indicates the (one standard deviation) uncertainty related to both the model and the noise of the target values.')

    res_en = st.radio("Plot Type", ('Predictions', 'Residuals')) == 'Residuals'

    # Sort data by x to prevent zigzag lines in Plotly
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    muN_sorted = muN[sort_idx]
    sigmaN_sorted = sigmaN[sort_idx]
    y_sorted = y[sort_idx]

    red = 'rgba(255, 0, 0, 0.3)'
    y_base = - res_en * y_sorted
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_sorted, y=muN_sorted - sigmaN_sorted + y_base,
        fill=None, mode='lines', line_color=red, showlegend=False))
    fig.add_trace(go.Scatter(x=x_sorted, y=muN_sorted + y_base,
        fill='tonexty', mode='lines', line_color='red', fillcolor=red, showlegend=False))
    fig.add_trace(go.Scatter(x=x_sorted, y=muN_sorted + sigmaN_sorted + y_base,
        fill='tonexty', mode='lines', line_color=red, fillcolor=red, showlegend=False))
    fig.add_trace(go.Scatter(x=x_sorted, y=y_sorted + y_base,
        mode='markers', marker_color='blue', showlegend=False))
        
    fig.update_layout(showlegend=False)
    fig.update_layout(xaxis_title=xlabel, yaxis_title=(r"$\Delta$ " if res_en else "") + ylabel)
    st.plotly_chart(fig, use_container_width=True)
    
    weights = pd.DataFrame({'Weights': df.columns[0:-1], 'Values': mN})
    
    # Clean LaTeX rendering
    eq_parts = []
    for j, label in enumerate(weights['Weights']):
        val = weights['Values'][j]
        clean_label = label.replace('_', '\_')
        eq_parts.append(f"{val:.4f} \\cdot {clean_label}")
    eq_str = f"$${ylabel} \\approx " + " + ".join(eq_parts) + "$$"
    st.markdown(eq_str)
    
    st.markdown(f"**Max Absolute Error ($|\Delta {ylabel}|$):** {np.max(np.abs(muN - y)):.4f}")
    
    st.markdown(get_table_download_link_csv(weights, 'Download weights as CSV.'), unsafe_allow_html=True)

st.write('---')

with st.expander("Learn More about Bayesian Regression"):
    st.markdown(r'''
    Bayesian regression is a very powerful framework at the origin of least-square and regularized least square fitting methods. The heart of the reasoning is the Bayes formula:
    $$P(W|Y) = \frac{P(Y|W)P(W)}{P(Y)}$$
    Conditioning every probabilities above on $X$ and $\mathcal{M}$ yields the main formula of interest:
    $$P(W|Y,X,\mathcal{M}) = \frac{P(Y|W,X,\mathcal{M})P(W|X,\mathcal{M})}{P(Y|X,\mathcal{M})}$$
    where $Y = [y_i]$, $X = [x_{i,j}]$ and $W = [w_j]$ are respectively the targets, the features and the weights matrices.
    
    Let us analyze each term:
    - $P(W|X,\mathcal{M})$ is the weight prior, which is taken as a 0-mean Gaussian distribution of precision parameter $\alpha$ herein (it does not depend on $X$).
    - $P(Y|W,X,\mathcal{M})$ is the likelihood, which in the light of the model $\mathcal{M}$ is a $XW$-mean diagonal Gaussian of precision $\beta$.
    - $P(Y|X,\mathcal{M})$ is the model evidence which indicates how likely the model was to generate the target samples, when marginalizing the weights.
    - $P(W|Y,X,\mathcal{M})$ is the weights posterior probability distribution which is what we want to infer.
    
    Maximizing (w.r.t. $W$) the probability $P(W|Y,X,\mathcal{M})$ yields the MAP weights. Marginalizing (w.r.t. $W$) the model prediction from $\mathcal{M}$ yields the measure of the uncertainty in the prediction.
    
    The $\alpha$ and $\beta$ hyperparameters can be set automatically by maximizing the model evidence. Refer to the following book for more details:
    :green_book: Bishop, Christopher. (2006). Pattern Recognition and Machine Learning. 10.1117/1.2819119. Page 152 to 171.
    ''')

st.write('Provided as is without any warranty.')
st.write('Created by Lutz Christophe - lutz.christophe@gmail.com - April 2021')
