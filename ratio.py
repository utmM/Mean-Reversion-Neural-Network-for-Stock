# -*- coding: utf-8 -*-
'''
1. Data Preparation
   ratio.py generates the input for the neural network from the raw stock prices

   The code finds poles on moving average(curve). From them, decides "peaks and bottoms" during the terms.
   Then, generates accelarations from each peak and bottom, which are the speeds of differences from the peaks and bottoms.
   The input data for the neural network consits of 25 days' differences of stock prices from moving averages and the accelaration at the closing as the example shows.(ex_input.csv)
   
! Disclaimer: The output(the input for neural network) possibly includes exceptions.
              Because of the "peaks and bottoms" are difined by artificiallly, mechanically, or judged by the algolism bellow shows for proccesing such a  big data in a short time, cases are that, there possibly includes some exceptions, which doesn't seems to be the peaks or bottoms in the term.
'''

from __future__ import print_function
from os import path
import operator as op
import numpy as np
import pandas as pd
# for releasing the limitation for calling the recursive function
import sys

# <<<<<<<<<<<<<<<<<<<<<<<< Define Constants >>>>>>>>>>>>>>>>>>>>>>>>>
T = 25 # Day of moving average
# Constants for finding the poles
Q = 10 # Q:term (to identify Rising(+) or Declining(-) phase of the moving average: the denominator of the probability）
q = 7 # q:threshold of the days (Standard for judging Rising(+) or Declining(-) phase: the numerator of the probability)

# <<<<<<<<<<<<<<<<<<<<<< Preparing the Row Data >>>>>>>>>>>>>>>>>>>>>>>>>>
def fetchCSV():
    if path.isfile('./index_N225.csv'):# access the same folder
        print('CSVあり fetch CSV for local: ' + './index_N225.csv')
        with open('./index_N225.csv') as f:
            f.read()
    
    all_data = pd.read_csv('index_N225.csv').set_index('Date').sort_index() # load as pandas dataframe sorting by days
    closing_data = pd.DataFrame()  # preparing pandas dataframe
    closing_data['N225'] = all_data['Close']  # set "Close" to 'N225' raw
    closing_data = closing_data.fillna(method='ffill') # fill the front
    return closing_data

# <<<<<<<<<<<<<<<<<<<<<< Load and Reshape the Data with Pandas >>>>>>>>>>>
# Culculate moving average and add to the dataframe
def add_divergence_data(closing_data):
    sum = float()
    closing_divergence_df = closing_data.assign(MEDIUM=0)
    for i in range(T-1, len(closing_divergence_df)):# （cf. range(*,**): not including **)
        for j in range(0, T):
            sum = sum + closing_divergence_df.iloc[i+j-(T-1),0]
        closing_divergence_df.iloc[i,1] = sum/T
        sum = 0
    return closing_divergence_df
# Get divergence only
def get_divergence_only(closing_divergence_df):
    divergence_df = closing_divergence_df.drop('N225', axis=1)
    delete_list = []
    for k in range(0, T-1):# cut initial 24 data
        delete_list.append(closing_divergence_df.index[k])
    divergence_df = divergence_df.drop(delete_list,axis=0)
    return divergence_df
# Add divergence of the moving average to the dataframe
def add_div_divergence(divergence_df):
    div_divergence_df = divergence_df.assign(DERIVATIVE=0)
    for l in range(1, len(div_divergence_df)):
        div_divergence_df.iloc[l,1] = div_divergence_df.iloc[l,0] - div_divergence_df.iloc[l-1,0]
    return div_divergence_df
# Add the operator(+or-) to the df (the divergence of the moving average df)
def add_sign_div_divergence(div_divergence_df):
    div_divergence_df = div_divergence_df.assign(SIGN=0)
    for i in range(1, len(div_divergence_df)):
        div_divergence_df.iloc[i,2] = np.sign(div_divergence_df.iloc[i,1])
    return div_divergence_df

# <<<<<<<<<<<<<<<<<<<<<< Recursive function: Algolithm for finding the poles >>>>>>>>>>>>>>>>>>>>>>>
# Parent method: (t: working day)
def find_poles_tTo2Q(t, div_divergence_df):
    # Cut the sequence of the SIGN(±) by the terms: t~Q and Q~2Q
    tToQ_df = div_divergence_df.iloc[t:t+Q, 2]
    QTo2Q_df = div_divergence_df.iloc[t+Q:t+2*Q, 2]
    print(tToQ_df,'tToQ_df')
    print(QTo2Q_df,'QTo2Q_df')
    
    # Children method:
    # (i) "Local Maximum" (!: the reason for using representetion "" is read in Disclaimer as above)
    if (tToQ_df >= 0).sum() >= q & (QTo2Q_df < 0).sum() >= q:
        print('MAX POLE FOUND')
        # Define POLE=1 if the day has the maximum absolute value of DERIVATIVE during the term: t~t+2Q
        tTo2Q_Ab_df = np.absolute(div_divergence_df.iloc[t:t+2*Q, 1])
        MAX_POLE_index = tTo2Q_Ab_df.idxmin()
        div_divergence_df.loc[MAX_POLE_index,'POLE'] = 1
        print(MAX_POLE_index, ': MAX_POLE_index')
        # Recursion if t+4Q < len(div_divergence_df), nor quit the method
        if t+4*Q >= len(div_divergence_df):
            return div_divergence_df
        else:
            print('MAX POLE ➔ recursion')
            return find_poles_tTo2Q(t+2*Q, div_divergence_df)
        
    # (ii) "Local Minimum"
    elif (tToQ_df < 0).sum() >= q & (QTo2Q_df >= 0).sum() >= q:
        print('MIN POLE FOUND')
        # Define POLE=-1 if the day has the minimum absolute value of DERIVATIVE during the term: t~t+2Q
        tTo2Q_Ab_df = np.absolute(div_divergence_df.iloc[t:t+2*Q, 1])
        MIN_POLE_index = tTo2Q_Ab_df.idxmin()
        div_divergence_df.loc[MIN_POLE_index,'POLE'] = -1
        print(MIN_POLE_index,': MIN_POLE_index')
        # Recursion if t+4Q < len(div_divergence_df), nor quit
        if t+4*Q >= len(div_divergence_df):
            return div_divergence_df
        else:
            print('MIN POLE ➔ recursion')
            return find_poles_tTo2Q(t+2*Q, div_divergence_df)

    # (iii) Neither Local MAX nor MIN
    else:# POLE=0
        # Recursion if t+2Q < len(div_divergence_df), nor quit
        if t+2*Q >= len(div_divergence_df):
            return div_divergence_df
        else:
            print('NO POLE ➔ recursion')
            return find_poles_tTo2Q(t+1, div_divergence_df)

# Amendment for the (possible) consecutive of local max or min
def modify_poles(div_divergence_df):
    poles_only_df = div_divergence_df[div_divergence_df.loc[:, 'POLE'] != 0]
    print(poles_only_df, ': POLES ONLY')
    for p in range(0, len(poles_only_df)-1):# compare sequencal poles
        if poles_only_df.iloc[p, 3] == poles_only_df.iloc[p+1, 3]:# The case of consecution
            if poles_only_df.iloc[p,3] == 1:# MAX POLE consecution case
                zero_index = poles_only_df.iloc[p:p+2,0].idxmin()# index(day), which has smaller N225 price
                div_divergence_df.loc[zero_index, 'POLE'] = 0# amend the POLE is 0
            elif poles_only_df.iloc[p,3] == -1:# MIN POLE consecution case
                zero_index = poles_only_df.iloc[p:p+2,0].idxmax()# indes(day), which has bigger N225 price
                div_divergence_df.loc[zero_index, 'POLE'] = 0# amend the POLE is 0
    return div_divergence_df

# Add N225 to the df
def add_N225(div_divergence_df, closing_data):
    div_divergence_df = pd.concat([div_divergence_df, closing_data], axis=1)
    # Delete the line including NaN
    div_divergence_df = div_divergence_df.dropna(how='any')
    return div_divergence_df

# Decide the peaks and bottoms of N225
def find_ceiling_bottom(div_divergence_df):
    div_divergence_df = div_divergence_df.assign(CEILING_BOTTOM=0)
    # Get peak <-> bottom to be alternately arranged
    poles_only_df = div_divergence_df[div_divergence_df.loc[:,'POLE'] != 0]
    print(poles_only_df, ': poles_only_df')
    for j in range(0,len(poles_only_df)-1):
        if poles_only_df.iloc[j,3] == -1:# If MIN POLE -> MAX POLE, the "peak" exists during the term (!: the reason for using representetion "" is read in Disclaimer as above)
            CEILING_index = div_divergence_df.loc[poles_only_df.index[j]:poles_only_df.index[j+1],'N225'].idxmax()
            div_divergence_df.loc[CEILING_index,'CEILING_BOTTOM'] = 1
        else:# If MAX POLE -> MIN POLE, the "bottom" exists during the term (!: the reason for using representetion "" is read in Disclaimer as above)
            BOTTOM_index = div_divergence_df.loc[poles_only_df.index[j]:poles_only_df.index[j+1],'N225'].idxmin()
            div_divergence_df.loc[BOTTOM_index,'CEILING_BOTTOM'] = -1
    # Caution: The head and tail of the data will be lost
    return div_divergence_df

# Normalize the peak and bottom
def add_normalized_ceiling_bottom(div_divergence_df):
    div_divergence_df = div_divergence_df.assign(C_B_DEGREE=0)
    ceiling_bottom_only_df = div_divergence_df[div_divergence_df.loc[:,'CEILING_BOTTOM'] !=0]
    for n in range(0,len(ceiling_bottom_only_df)-1):
        if ceiling_bottom_only_df.iloc[n,5] == 1:# Case: peak -> bottom
            ceiling_idx = ceiling_bottom_only_df.index[n]
            bottom_idx = ceiling_bottom_only_df.index[n+1]
            normalize_df = div_divergence_df.loc[ceiling_idx:bottom_idx,:]
            for k in range(0, len(normalize_df)):
                div_divergence_df.loc[normalize_df.index[k],'C_B_DEGREE'] = (normalize_df.iloc[k,4]-normalize_df.loc[bottom_idx,'N225'])/(normalize_df.loc[ceiling_idx,'N225']-normalize_df.loc[bottom_idx,'N225'])
        elif ceiling_bottom_only_df.iloc[n,5] == -1:# Case: bottom -> peak
            bottom_idx = ceiling_bottom_only_df.index[n]
            ceiling_idx = ceiling_bottom_only_df.index[n+1]
            normalize_df = div_divergence_df.loc[bottom_idx:ceiling_idx,:]
            for k in range(0,len(normalize_df)):
                div_divergence_df.loc[normalize_df.index[k],'C_B_DEGREE'] = (normalize_df.iloc[k,4]-normalize_df.loc[bottom_idx,'N225'])/(normalize_df.loc[ceiling_idx,'N225']-normalize_df.loc[bottom_idx,'N225'])
    return div_divergence_df
# So far, peak and bottom has been succecefully found.

# Culuculate and add the accelaration (UP_RATIO)
def add_up_ratio(div_divergence_df):
    div_divergence_df = div_divergence_df.assign(UP_RATIO=0)
    for i in range(0,len(div_divergence_df)-1):
        div_divergence_df.iloc[i+1,7] = div_divergence_df.iloc[i+1,6]-div_divergence_df.iloc[i,6]
    return div_divergence_df

# Add the differences of the price from moving averages
def add_difference(div_divergence_df):
    div_divergence_df = div_divergence_df.assign(RATE_OF_DIFFERENCE=0)
    for j in range(0, len(div_divergence_df)):
        div_divergence_df.iloc[j,8] = div_divergence_df.iloc[j,4]/div_divergence_df.iloc[j,0]-1
    return div_divergence_df

# <<<<<<<<<<<<<<<<<<<< Convert to input for the neural network >>>>>>>>>>>>>>>>
def get_input_df(div_divergence_df):
    # Add row name: last 0~24 days' differencial(DIFF:__(day) and the accelaration(UP_RATIO)
    col_list = []
    for i in range(0, T):
        col_name = 'DIFF:' + str((T-1)*-1+i)
        col_list.append(col_name)
    col_list.append('UP_RATIO')
    # Prepare the dataframe
    div_divergence_df = div_divergence_df[div_divergence_df.loc[:, 'UP_RATIO'] !=0] # Delete the line of no accelaration
    input_df = pd.DataFrame(index=div_divergence_df.index[T-1:], columns=col_list)
    # A: Add differencial of last 0~24 days【this takes the time】
    for k in range(0, len(input_df)):
        for l in range(0, T):
            input_df.iloc[k,l] = div_divergence_df.iloc[k+l,8]
        print('Now day:', input_df.index[k], 'Data: ', input_df.iloc[k,:],'has been generated.')
    # B: Add accelartions
    for m in range(0,len(input_df)):
        input_df.iloc[m,T] = div_divergence_df.iloc[T-1+m,7]
    return input_df


# ###################### The Main Function ####################################
# (Terminal setting) To display and check the processing data
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 10000)
# Data loading (Use N225 close only)
closing_data = fetchCSV()
print(closing_data)
# # for releasing the limitation for calling the recursive function (Manage Fatal Python error: Cannot recover from stack overflow.)
print(sys.getrecursionlimit())
sys.setrecursionlimit(15000)
print(sys.getrecursionlimit())
# Generate moving average dataframe
closing_divergence_df = add_divergence_data(closing_data)
#print('closing_divergence_df: ',closing_divergence_df)
# (Amend the data)
divergence_df = get_divergence_only(closing_divergence_df)
print(divergence_df, ': divergence_df')
# Add the difference of the moving average
div_divergence_df = add_div_divergence(divergence_df)
print(div_divergence_df, ': div_divergence_df')
# Add the operator(+or-) to the df (the divergence of the moving average df)
div_divergence_df = add_sign_div_divergence(div_divergence_df)
print(div_divergence_df,': ADDED SIGN')
# Find Maximum Pole and Minimum Pole
div_divergence_df = div_divergence_df.assign(POLE=0)
print(div_divergence_df, ': POLE=0 ADDED')
div_divergence_df = find_poles_tTo2Q(1, div_divergence_df)
print(div_divergence_df, ': POLES FOUND')
# Amendment for the (possible) consecutive of local max or min
div_divergence_df = modify_poles(div_divergence_df)
print(div_divergence_df, ': POLES MODIFIED')
# Add N225 and some amendment
div_divergence_df = add_N225(div_divergence_df, closing_data)
print(div_divergence_df, ': N225 ADDED')
# Decide the peaks and bottoms of N225
div_divergence_df = find_ceiling_bottom(div_divergence_df)
print(div_divergence_df, ': CEILING and BOTTOM FOUND')
# So far, peak and bottom has been succecefully found.
# Normalize
div_divergence_df = add_normalized_ceiling_bottom(div_divergence_df)
print(div_divergence_df, ': NORMALIZED CEILING and BOTTOM ADDED')

# Culuculate and add the accelaration (UP_RATIO)
div_divergence_df = add_up_ratio(div_divergence_df)
print(div_divergence_df, ': UP_RATIO ADDED')

# Add the differences of the price from moving averages
div_divergence_df = add_difference(div_divergence_df)
print(div_divergence_df, ': DIFFERENCEES ADDED')

# <<< Convert to input for the neural network >>>
input_df = get_input_df(div_divergence_df)
print(input_df, ': INPUT_DF GENERATED')
# The data for the Neural Network has been generated.

''' cf. this does't work for me. On terminal, copy & paste the output.
# Save and check
with open('input_df.csv', 'w') as f1:
    csv = input_df.to_csv
    f1.write(str(csv))
with open('input_df.csv') as f2:
    saved_csv = f2.read()
    print(saved_csv, ': THIS SAVED IN THE FILE')
'''

############### END OF THE CODE  ##########################################################
