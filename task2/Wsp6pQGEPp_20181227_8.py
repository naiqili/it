#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle

with open('/var/www/html/information_theory/feima/uploads/Wsp6pQGEPp_20181227_0.pkl', 'rb') as f:
    (stock_to_id, id_to_stock) = pickle.load(f)

m = 512 # num of stocks we use
_step = 0
step = 100
B = []
stock_state = np.zeros(m, dtype=int)

time_dict = {}
def time_to_id(time):    
    global time_dict
    if time not in time_dict.keys():
        time_dict[time] = len(time_dict.keys())
    return time_dict[time]

def build_mat(ss, history_stock_data):
    '''returns a m x s data matrix'''
    n = time_to_id(ss) + 1
    mat = np.ones((m, n))*-1
    df = history_stock_data
    for ind, row in df.iterrows():
        s = row['Stock Code']
        if s not in stock_to_id.keys():
            continue
        t = str(row['Time'])
        p = row['Opening Price']
        s_id = stock_to_id[s]
        t_id = time_to_id(t)
        if s_id < m and t_id < n:
            mat[s_id, t_id] = p
    return mat

def predict_next_x(P, index, w=5):
    t = index-1
    x = np.zeros(P.shape[1]) # x_t+1
    if index < w:
        w = index
    for k in range(m):
        cnt = 0
        for i in range(w):
            if P[t-i, k] > 0:
                x[k] += (P[t-i, k]/P[t, k])  # formula(1)
                cnt += 1
        if cnt > 0:
            x[k] /= cnt
    return x

def predict_next_b(B, P, mask, index, epsilon, w=5):
    t = index-1
    b_t = B[t]
    x_t1 = predict_next_x(P, index, w)
    x_mean = np.mean(x_t1)
    #print(x_t1, x_mean)
    lam = max(0.0, (epsilon-np.dot(b_t,x_t1))/(np.linalg.norm(x_t1-x_mean)**2))
    lam = min(100000, lam)
    #print(lam)
    #print(x_t1 - x_mean)
    b_t1 = b_t + lam * (x_t1 - x_mean)
    res = simplex_proj(b_t1)*mask 
    res *= 1/sum(res)
    return res #normalization

def simplex_proj(y):
    """ Projection of y onto simplex. """
    m = len(y)
    bget = False

    s = sorted(y, reverse=True)
    tmpsum = 0.

    for ii in range(m-1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1);
        if tmax >= s[ii+1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m-1] -1)/m
        
    return np.maximum(y-tmax,0.)

def update_state(money, mask, pv, bv):
    global stock_state
    #print(stock_state)
    old_state = stock_state
    all_money = money
    new_state = np.zeros(m, dtype=int)
    for i in range(m):
        if pv[i] > 0:
            new_state[i] = int(all_money * bv[i] / pv[i])
    buy_code, buy_num, sell_code, sell_num = [], [], [], []
    for i in range(m):
        if i not in id_to_stock.keys():
            continue
        if new_state[i] < old_state[i] and mask[i] == 1:
            sell_code.append(id_to_stock[i])
            sell_num.append(old_state[i] - new_state[i])
        if new_state[i] > old_state[i] and mask[i] == 1:
            buy_code.append(id_to_stock[i])
            buy_num.append(new_state[i] - old_state[i])
    stock_state = new_state
    return sell_code, sell_num, buy_code, buy_num

def invest(data_mat, money, mask, w=1, epsilon=1.00001):
    global B
    
    n = data_mat.shape[1]-1
    P = data_mat
    '''
    X = np.ones_like(P)
    for i in range(m):
        for j in range(1, n):
            X[i, j] = P[i, j] / P[i, j-1]
    '''
    P = P.transpose()
    #print(P)
            
    if n == 0:
        B.append(np.array([1/m for i in range(m)]))
    else:
        b = predict_next_b(B, P, mask, n, epsilon, w)
        B.append(b)
    #print('B:', B[-1])
    sell_code, sell_num, buy_code, buy_num = update_state(money, mask, P[-1], B[-1])
    return sell_code, sell_num, buy_code, buy_num

def get_avail(hist, tt):
    mask = np.zeros(512)
    for ind, row in hist.iterrows():
        s = row['Stock Code']
        t = row['Time']
        p = row['Opening Price']
        if s not in stock_to_id.keys():
            continue
        s_id = stock_to_id[s]
        if tt == t:
            mask[s_id] = 1
    return mask
    
def model(s, money, history_stock_data, investment_data):
    #path='/var/www/html/information_theory/feima/test_data.csv'
    global _step
    
    if _step > 0:
        _step -= 1
        add_data=pd.DataFrame(columns=['Time','Stocks you sell','Corresponding number of stocks you sell',
        'Stocks you buy','Corresponding number of stocks you buy']) 
        add_data=add_data.append({'Time': s}, ignore_index=True)
        B.append(B[-1])
        return add_data
    
    _step = step-1
        
    w=1200
    epsilon=1.00001
    history_stock_data = history_stock_data[-w*520:]
    ss = str(s)
    mask = get_avail(history_stock_data, s)
    data_mat = build_mat(ss, history_stock_data)
    sell_code, sell_num, buy_code, buy_num = invest(data_mat, money, mask, w=w, epsilon=epsilon)    
    
    add_data=pd.DataFrame(columns=['Time','Stocks you sell','Corresponding number of stocks you sell',
                                   'Stocks you buy','Corresponding number of stocks you buy']) 
                                  
    if len(sell_code) > 0 and len(buy_code) > 0:
        s1 = ', '.join(["%d"%(x) for x in sell_code])
        s2 = ', '.join(["%d"%(x) for x in sell_num])
        s3 = ', '.join(["%d"%(x) for x in buy_code])
        s4 = ', '.join(["%d"%(x) for x in buy_num])
        add_data=add_data.append({'Time': s,'Stocks you sell':s1,'Corresponding number of stocks you sell':s2,
                                  'Stocks you buy':s3,'Corresponding number of stocks you buy':s4}, ignore_index=True)    
    elif len(sell_code) > 0:
        s1 = ', '.join(["%d"%(x) for x in sell_code])
        s2 = ', '.join(["%d"%(x) for x in sell_num])
        add_data=add_data.append({'Time': s,'Stocks you sell':s1,'Corresponding number of stocks you sell':s2}, ignore_index=True)    
    elif len(buy_code) > 0:
        s1 = ', '.join(["%d"%(x) for x in buy_code])
        s2 = ', '.join(["%d"%(x) for x in buy_num])
        add_data=add_data.append({'Time': s,'Stocks you buy':s1,'Corresponding number of stocks you buy':s2}, ignore_index=True)    
    else:
        add_data=add_data.append({'Time': s}, ignore_index=True)

    return add_data