import torch

import sys
import os

sys.path.append('src')

from utils import load_pth_file, load_pth_model

from model_single import ComplexModel as model_single

from itertools import product

from inputs import integrate_exp_spectrum, ppm_to_hz
import os.path

import pickle
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import datetime
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from ipywidgets import widgets
from copy import deepcopy
from IPython.display import display, Math


def code_matrix(array,key):
    if len(array.shape)<2:
        array=array.reshape(1,-1)
    init=f'{key}='
    matrix = ''
    for row in array:
        try:
            for number in row:
                matrix += f'{number}&'
        except TypeError:
            matrix += f'{row}&'
        matrix = matrix[:-1] + r'\\'
    return init+r'\begin{bmatrix}'+matrix+r'\end{bmatrix}'


def print_matrix(array):
    """
    Отрисовка массива в виде матрицы
    """
    matrix = ''
    for row in array:
        try:
            for number in row:
                matrix += f'{number}&'
        except TypeError:
            matrix += f'{row}&'
        matrix = matrix[:-1] + r'\\'
    print(r'\begin{bmatrix}'+matrix+r'\end{bmatrix}')
    display(Math(r'\begin{bmatrix}'+matrix+r'\end{bmatrix}'))
    
    
def cos_value(x,y):
    tx=torch.norm(x)
    ty=torch.norm(y)
    if tx==0 or ty==0:
        return tx
    else:
        return -x@y/tx/ty
    
    
    
def optimization(model2,target, width=0):
    """
    Основная функция, которая оптимизирует начальное приближение метаболитов, переданное в model2, под сигнал target'
    Модель возвращает модель, метрику до оптимизации m1, метрика после оптимизации m2
    """ 
    
    """
    Подобный код закрепляет фиксирует параметры системы
    for m in model2.models:
        m.r0.requires_grad=False
        m.w0.requires_grad=False        
    """
    
    loss_fn = nn.MSELoss(reduction='sum')
    loss_fn2=cos_value
    loss = 1
    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-2)
    #scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.9)
    if width!=0:
        kernel_lorenz=0.5/width * 1 * ((0.5 * width) ** 2 / ((0.5 * width) ** 2 + torch.arange(-4*width,4*width+1) ** 2)).view(1,1,-1)
    kernel_gauss=0.1*torch.exp(-(torch.arange(-4*width,4*width+1))**2/width**2).view(1,1,-1)

    t=F.conv1d(target[:,1].clone().view(1,-1),kernel_lorenz,padding=width*4).squeeze() if width>0 else target[:,1]
    ids_nn=t>0
    loss_vals = []
    num_epochs=500
    # optimizer.param_groups[0]['lr']=1e-2
    m1=torch.log(1+loss_fn2(model2(target[:,0])[:,1], t))
    for epoch in range(num_epochs):
        if loss>-100.995:
            optimizer.zero_grad()
            output = model2(target[:,0])
            loss = torch.log(1+loss_fn2(output[:,1], t[:])) #+torch.log(loss_fn(output[ids_nn,1], t[ids_nn])) 
#             print('loss', loss.item())
            loss.backward(retain_graph=True)

            optimizer.step()
            #scheduler.step(loss)
            loss_vals.append(loss.item())
    m2=torch.log(1+loss_fn2(model2(target[:,0])[:,1], t))
    
    return model2,m1,m2
    
def optimization_reg(model2, target, lambda_reg=0.5, width=0):
    """
    Тот же метод, но с примером регуляризации
    """
    loss_fn = nn.MSELoss(reduction='sum')
    loss_fn2 = cos_value
    
    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-2)
    
    if width != 0:
        kernel_lorenz = 0.5 / width * 1 * ((0.5 * width) ** 2 / ((0.5 * width) ** 2 + torch.arange(-4*width, 4*width+1) ** 2)).view(1, 1, -1)
    kernel_gauss = 0.1 * torch.exp(-(torch.arange(-4*width, 4*width+1))**2 / width**2).view(1, 1, -1)
    
    t = (F.conv1d(target[:, 1].clone().view(1, -1), kernel_lorenz, padding=width*4).squeeze()
         if width > 0 else target[:, 1])
    
    ids_nn = t > 0
    loss_vals = []
    num_epochs = 500
    m1 = torch.log(1 + loss_fn2(model2(target[:, 0])[:, 1], t))
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model2(target[:, 0])
        
        loss_main = torch.log(1 + loss_fn2(output[:, 1], t))
        
        reg_loss = 0
        for m in model2.models:
            """
            Регуляризуем 01 и 10 элементы матрицы J
            """
            reg_loss += (m.J0[0, 1] ** 2) + (m.J0[1, 0] ** 2) 
        reg_loss = lambda_reg * reg_loss
        
        loss = loss_main + reg_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        
        loss_vals.append(loss.item())
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}")
    
    m2 = torch.log(1 + loss_fn2(model2(target[:, 0])[:, 1], t))
    
    return model2, m1, m2    


def save_results(model2, m1, m2, metabolite):
    with open(f'{metabolite}.html', encoding='utf-8') as inf:
        strhtml = inf.read()
    sspl = strhtml.rsplit('</div>')

    s_ad = f'''
      <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
      <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
      '''
    for k in model2.state_dict().keys():
        ki = r'proton\:config' if k == 'proton_config' else k
        latex_formula = code_matrix(model2.state_dict()[k], ki)
        s_ad += f'''
      <div style="font-size: 24px;">
        \\[ {latex_formula} \\]
      </div>
      '''
    latex_formula = r"metric\:before\:" + f"= {m1}, after = {m2}"
    s_ad += f'''<div style="font-size: 24px;">
        \\[ {latex_formula} \\]
      </div>'''

    s = ''
    for i in range(len(sspl) - 1):
        s += sspl[i] + '</div>'

    s += s_ad
    s += sspl[-1]

    with open(f'{metabolite}.html', "w", encoding='utf-8') as outf:
        outf.write(s)
        
        
        
def prepare_html(model2,m1,m2,target, metabolite):
    f=go.Figure(layout={'xaxis':{'title':'Hz'}})
    f.add_trace(go.Scatter(x=target[:,0].numpy(),y=target[:,1].detach().numpy(),name='target'))
    f.add_trace(go.Scatter(x=target[:,0].numpy(),y=model2(target[:,0])[:,1].detach().numpy(),name='predicted'))
    # f.show()
    f.write_html(f'{metabolite}.html')
    save_results(model2,m1,m2,metabolite)
    #torch.save(model2.state_dict(),f'data/pth/{metabolite}.pth')
    
    
    
def optimize_and_save(metabolite):
    model2 = load_pth_model(f'data/pth/{metabolite}.pth')
    if model2 is None:
        return 0,0,0
    exp_spectrum=pd.read_csv(f'data/csv/{file}').values 
    ids=np.where(exp_spectrum[:,1]>1e-4)[0][[0,-1]]
    ids[0]=max(ids[0]-(ids[1]-ids[0])//2,0)
    ids[1]=min(ids[1]+(ids[1]-ids[0])//2,exp_spectrum.shape[0]-1)
    exp_spectrum=exp_spectrum[ids[0]:ids[1],:]
    exp_Hz=ppm_to_hz(exp_spectrum[:,0], model2.models[0].freq.numpy()).view(-1)
    exp_I=exp_spectrum[:,1]
    exp_area = integrate_exp_spectrum(exp_I, exp_Hz.view(-1).detach().numpy())
    exp_I = [i/exp_area for i in exp_I]
    tens_Hz = torch.FloatTensor(exp_Hz)
    tens_I = torch.FloatTensor(exp_I)

    target = torch.stack((tens_Hz, tens_I), -1).detach()
    model2,m1,m2=optimization(model2,target)
    prepare_html(model2,m1,m2,target, metabolite)
    return m1,m2,model2
