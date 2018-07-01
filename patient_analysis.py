# -*- coding: utf-8 -*-
"""
Holmusk

The task is to analyze the clinical and financial data of patients hospitalized for a certain condition.
It is attached with this email. Some variable names and patient_id's have been anonymized in this dataset.
You are required to join the data given in different tables, and find insights about the drivers of cost of care.

Cathey Wang
06/25/2018
"""

import math
import pandas as pd
import numpy as np
from datetime import datetime
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K


"""
Read csv
"""
def load_data(datadir):
    bill_amount_csv = datadir + "bill_amount.csv"
    bill_id_csv = datadir + "bill_id.csv"
    clinical_csv = datadir + "clinical_data.csv"
    demographics_csv = datadir + "demographics.csv"
    bill_amount = pd.read_csv(bill_amount_csv)
    bill_id = pd.read_csv(bill_id_csv)
    clinical = pd.read_csv(clinical_csv)
    demographics = pd.read_csv(demographics_csv)
    return bill_amount, bill_id, clinical, demographics


"""
generate X & Y for model fitting
X: clinical & demographical data
    sex, age, height, weight, race, residence, history 1:7, medication 1:6, symptom 1:5, lab 1:3
Y: sum cost of care of each individual visit, treat the same patient as separate at diff times
v1: use 0.5 for nan's - mid point for linear regression
"""
def gen_XY(bill_amount, bill_id, clinical, demographics):
    N = clinical.shape[0]
    X = np.zeros((N, 33))
    Y = np.zeros((N, 1))
    # use date of admission for age calc & bill referal
    print("generating data")
    for n in range(N):
        patient_id = clinical["id"][n]
        date_admit_str = clinical["date_of_admission"][n]
        bill_idx = bill_id.loc[(bill_id['patient_id'] == patient_id) & (bill_id["date_of_admission"] == date_admit_str)].index
        # calc total bill
        for i in bill_idx:
            bill = bill_id["bill_id"][i]
            amount_idx = bill_amount.loc[bill_amount['bill_id'] == bill].index
            amount_idx = amount_idx[0]      # guaranteed unique
            Y[n] += bill_amount["amount"][amount_idx]
        
        # calc age
        date_admit = datetime.strptime(date_admit_str, '%Y-%m-%d')
        dem_idx = demographics.loc[demographics['patient_id'] == patient_id].index
        dem_idx = dem_idx[0]
        date_birth_str = demographics["date_of_birth"][dem_idx]
        date_birth = datetime.strptime(date_birth_str, '%Y-%m-%d')
        age = date_admit.year - date_birth.year - ((date_admit.month, date_admit.day) < (date_birth.month, date_birth.day))
        
        # read basic info
        gender = demographics["gender"][dem_idx]
        if gender == "Female" or gender == "f":
            gender = 1
        else:   # "Male" or "m"
            gender = 0
        height = clinical["height"][n]
        weight = clinical["weight"][n]
        X[n][0:4] = [gender, age, height, weight]
        
        # read race & residence, x[n][4:11]
        race = demographics["race"][dem_idx]
        if race == "Indian" or race == "India":
            X[n][4] = 1
        elif race == "Chinese" or race == "chinese":
            X[n][5] = 1
        elif race == "Malay":
            X[n][6] = 1
        else:   # "Others"
            X[n][7] = 1
        residence = demographics["resident_status"][dem_idx]
        if residence == "Singaporean" or residence == "Singaporean citizen":
            X[n][8] = 1
        elif residence == "PR":
            X[n][9] = 1
        else:   # foreigner
            X[n][10] = 1
        
        # read history 1:7, 2&5 has nan, 3 has string
        history_2 = clinical["medical_history_2"][n]
        if np.isnan(history_2):
            history_2 = 0.5     # v1: use 0.5 for mid-point
        history_5 = clinical["medical_history_5"][n]
        if np.isnan(history_5):
            history_5 = 0.5     # v1: use 0.5 for mid-point
        history_3 = clinical["medical_history_3"][n]
        if history_3 == '0' or history_3 == 'No':
            history_3 = 0
        else:   # "1" or "Yes"
            history_3 = 1
        history_1 = clinical["medical_history_1"][n]
        history_4 = clinical["medical_history_4"][n]
        history_6 = clinical["medical_history_6"][n]
        history_7 = clinical["medical_history_7"][n]
        X[n][11:18] = [history_1, history_2, history_3, history_4, history_5, history_6, history_7]
        
        # read preop medication 1-6
        medication_1 = clinical["preop_medication_1"][n]
        medication_2 = clinical["preop_medication_2"][n]
        medication_3 = clinical["preop_medication_3"][n]
        medication_4 = clinical["preop_medication_4"][n]
        medication_5 = clinical["preop_medication_5"][n]
        medication_6 = clinical["preop_medication_6"][n]
        X[n][18:24] = [medication_1, medication_2, medication_3, medication_4, medication_5, medication_6]
        
        # read preop medication 1-6
        symptom_1 = clinical["symptom_1"][n]
        symptom_2 = clinical["symptom_2"][n]
        symptom_3 = clinical["symptom_3"][n]
        symptom_4 = clinical["symptom_4"][n]
        symptom_5 = clinical["symptom_5"][n]
        X[n][24:29] = [symptom_1, symptom_2, symptom_3, symptom_4, symptom_5]
        
        # read lab results 1-3
        lab_1 = clinical["lab_result_1"][n]
        lab_2 = clinical["lab_result_2"][n]
        lab_3 = clinical["lab_result_3"][n]
        X[n][29:32] = [lab_1, lab_2, lab_3]
        
        # additional params
        X[n][32] = weight/pow(height/100, 2)    # BMI
        
    print("done generating")
    return X, Y
    

"""
generate X & Y for model fitting
X: clinical & demographical data, specific of a certain cohort, eg. gender
Y: sum cost of care of each individual visit, treat the same patient as separate at diff times
"""
def separate_gender(X, Y):
    N, L = X.shape
    N_f, N_m = 0, 0
    X_f = np.zeros((N, L-1))
    X_m = np.zeros((N, L-1))
    Y_f = np.zeros((N, 1))
    Y_m = np.zeros((N, 1))
    for n in range(N):
        if X[n][0] == 1:    # female
            X_f[N_f][:] = X[n][1:L]
            Y_f[N_f] = Y[n]
            N_f += 1
        else:               # male
            X_m[N_m][:] = X[n][1:L]
            Y_m[N_m] = Y[n]
            N_m += 1
            
    X_f = X_f[0:N_f][:]
    X_m = X_m[0:N_m][:]
    Y_f = Y_f[0:N_f]
    Y_m = Y_m[0:N_m]
    return X_f, Y_f, X_m, Y_m


def separate_residence(X, Y):
    N, L = X.shape
    N_S, N_P, N_F = 0, 0, 0
    X_S = np.zeros((N, L-3))
    X_P = np.zeros((N, L-3))
    X_F = np.zeros((N, L-3))
    Y_S = np.zeros((N, 1))
    Y_P = np.zeros((N, 1))
    Y_F = np.zeros((N, 1))
    for n in range(N):
        if X[n][8] == 1:    # Singaporean
            X_S[N_S][0:8] = X[n][0:8]
            X_S[N_S][8:-1] = X[n][11:-1]
            Y_S[N_S] = Y[n]
            N_S += 1
        elif X[n][9] == 1:  # PR
            X_P[N_P][0:8] = X[n][0:8]
            X_P[N_P][8:-1] = X[n][11:-1]
            Y_P[N_P] = Y[n]
            N_P += 1
        elif X[n][10] == 1:  # Foreigners
            X_F[N_F][0:8] = X[n][0:8]
            X_F[N_F][8:-1] = X[n][11:-1]
            Y_F[N_F] = Y[n]
            N_F += 1

    X_S = X_S[0:N_S][:]
    X_P = X_P[0:N_P][:]
    X_F = X_F[0:N_F][:]
    Y_S = Y_S[0:N_S]
    Y_P = Y_P[0:N_P]
    Y_F = Y_F[0:N_F]
    return X_S, Y_S, X_P, Y_P, X_F, Y_F


def separate_race(X, Y):
    N, L = X.shape
    N_I, N_C, N_M, N_O = 0, 0, 0, 0
    X_I = np.zeros((N, L-4))
    X_C = np.zeros((N, L-4))
    X_M = np.zeros((N, L-4))
    X_O = np.zeros((N, L-4))
    Y_I = np.zeros((N, 1))
    Y_C = np.zeros((N, 1))
    Y_M = np.zeros((N, 1))
    Y_O = np.zeros((N, 1))
    for n in range(N):
        if X[n][4] == 1:    # indian
            X_I[N_I][0:4] = X[n][0:4]
            X_I[N_I][4:-1] = X[n][8:-1]
            Y_I[N_I] = Y[n]
            N_I += 1
        elif X[n][5] == 1:  # chinese
            X_C[N_C][0:4] = X[n][0:4]
            X_C[N_C][4:-1] = X[n][8:-1]
            Y_C[N_C] = Y[n]
            N_C += 1
        elif X[n][6] == 1:  # malaysian
            X_M[N_M][0:4] = X[n][0:4]
            X_M[N_M][4:-1] = X[n][8:-1]
            Y_M[N_M] = Y[n]
            N_M += 1
        elif X[n][7] == 1:  # other
            X_O[N_O][0:4] = X[n][0:4]
            X_O[N_O][4:-1] = X[n][8:-1]
            Y_O[N_O] = Y[n]
            N_O += 1

    X_I = X_I[0:N_I][:]
    X_C = X_C[0:N_C][:]
    X_M = X_M[0:N_M][:]
    X_O = X_O[0:N_O][:]
    Y_I = Y_I[0:N_I]
    Y_C = Y_C[0:N_C]
    Y_M = Y_M[0:N_M]
    Y_O = Y_O[0:N_O]
    return X_I, Y_I, X_C, Y_C, X_M, Y_M, X_O, Y_O


def separate_binary(X, Y):
    N, L = X.shape
    N_1, N_0 = 0, 0
    X_1 = np.zeros((N, L-1))
    X_0 = np.zeros((N, L-1))
    Y_1 = np.zeros((N, 1))
    Y_0 = np.zeros((N, 1))
    idx = 26
    for n in range(N):
        if X[n][idx] == 1:    # positive
            X_1[N_1][0:idx] = X[n][0:idx]
            X_1[N_1][idx:-1] = X[n][idx+1:-1]
            Y_1[N_1] = Y[n]
            N_1 += 1
        else:               # negative
            X_0[N_0][0:idx] = X[n][0:idx]
            X_0[N_0][idx:-1] = X[n][idx+1:-1]
            Y_0[N_0] = Y[n]
            N_0 += 1

    X_1 = X_1[0:N_1][:]
    X_0 = X_0[0:N_0][:]
    Y_1 = Y_1[0:N_1]
    Y_0 = Y_0[0:N_0]
    return X_1, Y_1, X_0, Y_0


"""
v1: Linear Regression based on 27 params
"""
def gen_datesets(X, Y):
    N,L = X.shape
    N_train = int(N*0.75)
    N_test = N-N_train
    
    X_train = np.zeros((N_train, L))
    X_test = np.zeros((N_test, L))
    Y_train = np.zeros((N_train, 1))
    Y_test = np.zeros((N_test, 1))
    
    idx = [i for i in range(N)]
    random.seed(3)
    random.shuffle(idx)
    thresh = int(np.ceil(N/4))
    for i in range(N):
        if i < thresh:
            X_test[i,:] = X[idx[i], :]
            #Y_test[i] = Y[idx[i]]
            Y_test[i] = math.log(Y[idx[i]])
        else:
            X_train[i-thresh,:] = X[idx[i], :]
            #Y_train[i-thresh] = Y[idx[i]]
            Y_train[i-thresh] = math.log(Y[idx[i]])
    
    return X_train, Y_train, X_test, Y_test
    
    
"""
v1: Linear Regression based on 27 params
"""
def fit_curve(X_train, Y_train, X_test, Y_test, f):
    reg = linear_model.LinearRegression()
    reg.fit(X_train,Y_train)
    #coeffs = reg.coef_
    Y_train_pred = reg.predict(X_train)
    Y_train = np.exp(Y_train)
    Y_train_pred = np.exp(Y_train_pred)
    plt.figure(f)
    plt.hold(True)
    plt.plot(Y_train, Y_train, 'k')
    plt.scatter(Y_train, Y_train_pred)
    plt.xlabel('Y_truth')
    plt.ylabel('Y_pred')
    plt.hold(False)
    print("Training:\nMean squared error: %.2f"
      % mean_squared_error(Y_train, Y_train_pred))
    print("R2 score: %.2f"
      % r2_score(Y_train, Y_train_pred))
    
    Y_test_pred = reg.predict(X_test)
    Y_test = np.exp(Y_test)
    Y_test_pred = np.exp(Y_test_pred)
#    plt.figure(f+1)
#    plt.hold(True)
#    plt.plot(Y_test, Y_test, 'k')
#    plt.scatter(Y_test, Y_test_pred)
#    plt.xlabel('Y_truth')
#    plt.ylabel('Y_pred')
#    plt.hold(False)
    print("Testing:\nMean squared error: %.2f"
      % mean_squared_error(Y_test, Y_test_pred))
    print("R2 score: %.2f"
      % r2_score(Y_test, Y_test_pred))
    
    return reg, Y_train_pred, Y_test_pred


"""
Neural net
"""
def ann_model():

    model = Sequential()
    model.add(Dense(20, input_dim=32, kernel_initializer='lecun_normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='lecun_normal', activation = 'relu'))
    model.add(Dense(1, kernel_initializer='lecun_normal', activation = 'linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model



"""
Train model
"""
def train_model(X_train, Y_train, X_test, Y_test):
    batch_size = 200
    epochs = 400
    
    # Load model
    model = ann_model()

    # Train model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                   epochs = epochs, shuffle = True)#, callbacks = callbacks)

    return model


"""
Evaluate model performance
"""
def eval_model(model, X_train, Y_train, X_test, Y_test):
    Y_train_pred = model.predict(X_train)
    plt.figure(0)
    plt.hold(True)
    plt.plot(Y_train, Y_train, 'k')
    plt.scatter(Y_train, Y_train_pred)
    plt.xlabel('Y_truth')
    plt.ylabel('Y_pred')
    plt.hold(False)
    print("Training:\nMean squared error: %.2f"
      % mean_squared_error(Y_train, Y_train_pred))
    print("R2 score: %.2f"
      % r2_score(Y_train, Y_train_pred))
    
    Y_test_pred = model.predict(X_test)
    plt.figure(1)
    plt.hold(True)
    plt.plot(Y_test, Y_test, 'k')
    plt.scatter(Y_test, Y_test_pred)
    plt.xlabel('Y_truth')
    plt.ylabel('Y_pred')
    plt.hold(False)
    print("Testing:\nMean squared error: %.2f"
      % mean_squared_error(Y_test, Y_test_pred))
    print("R2 score: %.2f"
      % r2_score(Y_test, Y_test_pred))

    return Y_train_pred, Y_test_pred


"""
Driver
"""
if __name__ == "__main__":
    datadir = "../data/"
    bill_amount, bill_id, clinical, demographics = load_data(datadir)
    X, Y = gen_XY(bill_amount, bill_id, clinical, demographics)
#    X_f, Y_f, X_m, Y_m = separate_gender(X, Y)
    X_S, Y_S, X_P, Y_P, X_F, Y_F = separate_residence(X, Y)
    X_1, Y_1, X_0, Y_0 = separate_binary(X_F, Y_F)
#    X_I, Y_I, X_C, Y_C, X_M, Y_M, X_O, Y_O = separate_race(X_F, Y_F)

    X_train, Y_train, X_test, Y_test = gen_datesets(X_1, Y_1)
    reg, Y_train_pred, Y_test_pred = fit_curve(X_train, Y_train, X_test, Y_test, 0)
    
    X_train, Y_train, X_test, Y_test = gen_datesets(X_0, Y_0)
    reg, Y_train_pred, Y_test_pred = fit_curve(X_train, Y_train, X_test, Y_test, 2)
    
    #model = train_model(X_train, Y_train, X_test, Y_test)
    #print("Generating performance...")
    #Y_train_pred, Y_test_pred = eval_model(model, X_train, Y_train, X_test, Y_test)