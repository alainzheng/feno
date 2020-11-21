# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:03:32 2020

@author: Alain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,leastsq,least_squares
import time
import datetime
from sklearn.linear_model import LinearRegression
np.set_printoptions(suppress=True) # for no scientific notation


def non_linear_regression_Ce(Vdot, Cw, Dno, Calv):
    return Cw*( 1-np.exp(-Dno/Vdot) ) + Calv*np.exp(-Dno/Vdot)


def model(Vdot, Cw, Dno, Calv):
    exponent = np.array(-Dno/Vdot, dtype= np.float32)
    return Cw*( 1-np.exp(exponent) ) + Calv*np.exp(exponent)



if __name__ == '__main__':

    tic = time.perf_counter()

    print('-'*30)
    print('running feno.py ...')
    print('-'*30)
    
    # load data from excel file
    df = pd.read_excel('feno_subjects.xlsx',header = None) 
    df = np.array(df) 
    # print(df)

    # separate into common variables
    #############################################
    #######          Parameters          ########
    #############################################
    
    Vdot = df[0,1:]  # 9 flow rates in ml/s
    VdotSteps = np.arange(Vdot[0], Vdot[-1])
    tablePatient = df[1:11,1:]  # 10 patients x 9 flow rates ppb
    tableMean = df[11,1:] # mean values per flow rate
    tableSem = df[12,1:] # sem values per flow rate
    
    #############################################
    #######   2point linear regression   ########
    #############################################
    
    qmean = tableMean[[3,5]]*Vdot[[3,5]]
    reg = LinearRegression().fit(tableMean[[3,5]][:,np.newaxis], qmean)
    reg.coef_
    reg.intercept_
    
    fig, ax = plt.subplots()
    ax.scatter(tableMean[[3,5]][:,np.newaxis], qmean, color='r',label='raw data 6')
    ax.plot(tableMean[[3,5]][:,np.newaxis], (reg.coef_[0]*tableMean[[3,5]]) + reg.intercept_,
            color='b',label='linear regression 6' )
    ax.set_title('Expired NO concentration (Ce) as a function of expiratory flow (Vdot) \n Non linear regression on mean data')
    ax.set_xlabel('Ce: exhaled NO concentration')
    ax.set_ylabel('Qdot: Quantity of NO exhaled per unit time')
    ax.grid()
    ax.legend()
    plt.show()

    Dno = []
    Cw = []
    patient=0
    for patient in range(10): # mean also taken 
        reg_lin =  LinearRegression().fit(tablePatient[patient][[3,5]][:,np.newaxis],
                                         tablePatient[patient][[3,5]]*Vdot[[3,5]])
        Dno.append(reg_lin.coef_[0])
        absciss = -reg_lin.intercept_ / reg_lin.coef_[0]
        Cw.append(absciss)
        
    lrParams = []
    lrParams.append(Dno)
    lrParams.append(Cw)
    lrParams = np.abs(np.swapaxes(np.array(lrParams), axis1=0, axis2=1))
    
    print(pd.DataFrame(np.fliplr(lrParams).round(2),
                        index = ['patient %d'%x for x in range(1,11)],
                        columns = ['Dno', 'Cw']))    
    
    
    #############################################
    #######   6point linear regression   ########
    #############################################
    """
    qmean = tableMean[:6]*Vdot[:6]
    reg = LinearRegression().fit(tableMean[:6][:,np.newaxis], qmean)
    reg.coef_
    reg.intercept_
    
    fig, ax = plt.subplots()
    ax.scatter(tableMean[:6][:,np.newaxis], qmean, color='r',label='raw data 6')
    ax.plot(tableMean[:6][:,np.newaxis], (reg.coef_[0]*tableMean[:6]) + reg.intercept_,
            color='b',label='linear regression 6' )
    ax.set_title('Expired NO concentration (Ce) as a function of expiratory flow (Vdot) \n Non linear regression on mean data')
    # ax.set_xlabel('Flow (mL/s)')
    # ax.set_ylabel('Expired NO concentration (ppb)')
    ax.grid()
    ax.legend()
    plt.show()

    Dno = []
    Cw = []
    patient=0
    for patient in range(10): # mean also taken 
        reg_lin =  LinearRegression().fit(tablePatient[patient][:6][:,np.newaxis],
                                         tablePatient[patient][:6]*Vdot[:6])
        Dno.append(reg_lin.coef_[0])
        absciss = -reg_lin.intercept_ / reg_lin.coef_[0]
        Cw.append(absciss)
        
    lrParams = []
    lrParams.append(Dno)
    lrParams.append(Cw)
    lrParams = np.abs(np.swapaxes(np.array(lrParams), axis1=0, axis2=1))
    
    print(pd.DataFrame(np.fliplr(lrParams).round(2),
                        index = ['patient %d'%x for x in range(1,11)],
                        columns = ['Dno', 'Cw']))    

    """

    #############################################
    ####### 9point non linear regression ########
    #############################################
    """
    print('-'*30)
    print('9 points non linear regression')
    print('-'*30)
    ##### this function returns the values of the 3 parameters by fitting them with the data from table mean
    meanNlrParams, meanNlrCov = curve_fit(non_linear_regression_Ce, Vdot, tableMean)
    
    fig, ax = plt.subplots()
    ax.scatter(Vdot, tableMean,  label='Raw mean data')
    ax.plot(VdotSteps, model(VdotSteps, *meanNlrParams), color='r',label='Non linear regression method')
    ax.set_title('Expired NO concentration (Ce) as a function of expiratory flow (Vdot) \n Non linear regression on mean data')
    ax.set_xlabel('Flow (mL/s) \n Figure 2 ')
    ax.set_ylabel('Expired NO concentration (ppb)')
    ax.grid()
    ax.legend()
    ax.annotate(' Cw = %.2f ppb \n Dno = %.2f nL/s/ppb x 10^-3 \n Calv =  %.2f ppb'
                %(meanNlrParams[0], meanNlrParams[1], meanNlrParams[2]),
                xy=(200, 160), xycoords='figure pixels')
    fig.savefig('9_nlr_mean.png')
    plt.show()
    
    ##### compute the parameters for every patient
    nlrParams = np.zeros((10, 3))
    nlrCov = np.zeros((10, 3, 3))
    for patient in range(10): # mean also taken 
        nlrParams[patient], nlrCov[patient] = curve_fit(f = non_linear_regression_Ce,
                                                        xdata = Vdot,
                                                        ydata = tablePatient[patient])
        if patient == 5:
            print('Reminder: Patient 6 parameters do not converge')
    print(pd.DataFrame(np.fliplr(nlrParams).round(2),
                        index = ['patient %d'%x for x in range(1,11)],
                        columns = ['Calv', 'Dno', 'Cw']))
    nlrParams = np.delete(nlrParams,5, axis=0)
        
    nlrParamsGeoMean = np.exp(np.sum(np.log(nlrParams), axis=0)/(len(nlrParams))) # careful here divide by 9 instead of 10
    nlrParamsSem = nlrParamsGeoMean * np.std(np.log(nlrParams), axis=0)/np.sqrt(len(nlrParams))     # here we use 9 as n but in the paper they used n=8 error?

    print('-'*30)
    print('Mean values: \nCalv = %.2f ppb \nDno = %.2f nL/s/ppb x 10^-3 \nCw =  %.2f ppb'%(nlrParamsGeoMean[2],
                                                                                           nlrParamsGeoMean[1],
                                                                                           nlrParamsGeoMean[0]))
    print('-'*30)
    print('SEM values: \nCalv = %.2f ppb \nDno = %.2f nL/s/ppb x 10^-3 \nCw =  %.2f ppb'%(nlrParamsSem[2],
                                                                                           nlrParamsSem[1],
                                                                                           nlrParamsSem[0]))
    print('-'*30)
    
    #### verifier les unités
    
    Calv = meanNlrParams[2]
    Dno = meanNlrParams[1]
    Cw = meanNlrParams[0]
    qdot = tableMean*Vdot*10**-3  # ppb*ml/s *10**-3
    maxNO = Dno*Cw*10**-3
    
    maxNO_graph = np.array([maxNO]*len(Vdot))
    """
    """"
    ########## figure 3A
    fig, ax = plt.subplots()
    ax.scatter(Vdot, qdot, label = 'qdot, raw data')
    ax.plot(VdotSteps, model(VdotSteps,*meanNlrParams)*VdotSteps*10**-3, color='r', label = 'qdot, nonlinear regression')
    ax.plot(Vdot, qdot - (Vdot*Calv*10**(-3)), label = 'q_D')
    plt.plot(Vdot,maxNO_graph, '--',label = 'Dno*Cw')
    ax.grid()
    plt.ylim(0,7)
    plt.xlim(0,1000)
    ax.set_title('Plot of NO output (q) and (q_D) against expiratory flow rates (0 and 1.0 L/s).')
    ax.set_xlabel('Flow (mL/s) \n Figure 3')
    ax.set_ylabel('Expired NO output (nL/s)')
    ax.legend()
    plt.show()


    ########## figure 3B
    fig, ax = plt.subplots()
    ax.scatter(Vdot, qdot, label = 'qdot, raw data')
    ax.plot(VdotSteps, model(VdotSteps, *meanNlrParams)*VdotSteps*10**-3, color='r', label = 'qdot, nonlinear regression')
    ax.plot(Vdot, qdot - (Vdot*Calv*10**(-3)), label = 'q_D')
    plt.plot(Vdot, maxNO_graph, '--', label = 'Dno*Cw')
    ax.grid()
    plt.ylim(0,1)
    plt.xlim(0,20)
    ax.set_title('Plot of NO output (q) and (q_D) against expiratory flow rates (0 and 1.0 L/s).')
    ax.set_xlabel('Flow (mL/s) \n Figure 3')
    ax.set_ylabel('Expired NO output (nL/s)')
    ax.legend()
    plt.show()
    """
    
    
    
    
    
    
    
    
    ########################################
    #############    TIMER     #############
    ########################################
    
    toc = time.perf_counter()
    print('\n toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
      
