# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:03:32 2020

Airway NO Diffusion in Asthma - Role in the Pulmonary function and Bronchial Responsiveness 
Article authors: PHILIP E. SILKOFF, JIMMIE T. SYLVESTER, NOE ZAMEL, and SOLBERT PERMUTT

Code Author: Mathieu BIAVA, Philippe QUESTEL, Alain Kai Rui ZHENG 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import datetime
from sklearn.linear_model import LinearRegression
np.set_printoptions(suppress=True) # for no scientific notation



##### expression of Ce 
def non_linear_regression_Ce(Vdot, Cw, Dno, Calv):
    return Cw*( 1-np.exp(-Dno/Vdot) ) + Calv*np.exp(-Dno/Vdot)

##### the non linear model ->parameters values  are here known
def nlr_model(Vdot, Cw, Dno, Calv):
    exponent = np.array(-Dno/Vdot, dtype= np.float32)
    return Cw*( 1-np.exp(exponent) ) + Calv*np.exp(exponent)




def points9_non_linear_regression():
    
    print('*'*71)
    print('*'*20, '9 point non-linear regression', '*'*20)
    print('*'*71)
    

    
    ##### compute the parameters for every patient
    nlrParams = np.zeros((10, 3))
    nlrCov = np.zeros((10, 3, 3))
    for patient in range(10): # mean also taken 
        nlrParams[patient], nlrCov[patient] = curve_fit(f = non_linear_regression_Ce,
                                                        xdata = Vdot,
                                                        ydata = PatientsCe[patient])
        if patient == 5:
            print('Reminder: Patient 6 parameters do not converge')
            
            
    print(pd.DataFrame(np.fliplr(nlrParams).round(2),
                        index = ['patient %d'%x for x in range(1,11)],
                        columns = ['Calv (ppb)', 'Dno (nL/s/ppb x 10^-3)', 'Cw (ppb)']))
    nlrParams = np.delete(nlrParams,5, axis=0)
    
    nlrParamsGeoMean = np.exp(np.sum(np.log(nlrParams), axis=0)/(len(nlrParams))) # careful here divide by 9 instead of 10
    nlrParamsSem = nlrParamsGeoMean * np.std(np.log(nlrParams), axis=0)/np.sqrt(len(nlrParams))     # here we use 9 as n but in the paper they used n=8 error?

    print('-'*30)
    print('Mean values for 9point non-linear: \nCalv = %.2f ppb \nDno = %.2f nL/s/ppb x 10^-3 \nCw =  %.2f ppb'%(nlrParamsGeoMean[2],
                                                                                           nlrParamsGeoMean[1],
                                                                                           nlrParamsGeoMean[0]))
    print('-'*30)
    print('SEM values for 9point non-linear: \nCalv = %.2f ppb \nDno = %.2f nL/s/ppb x 10^-3 \nCw =  %.2f ppb'%(nlrParamsSem[2],
                                                                                           nlrParamsSem[1],
                                                                                           nlrParamsSem[0]))
    print('-'*30)
    
    
    #############################################
    #######            FIGURES            #######
    #############################################
    
    ##### meanNlrParams are the optimal values for the parameters Cw, Dno, Calv
    meanNlrParams, meanNlrCov = curve_fit(f = non_linear_regression_Ce,
                                          xdata = Vdot,
                                          ydata = MeanCe)
    
    Calv = meanNlrParams[2] 
    Dno = meanNlrParams[1] 
    Cw = meanNlrParams[0]
    qdot = MeanCe*Vdot    # =nlr_model(Vdot,*meanNlrParams) * Vdot
    DnoCw = np.array([Dno*Cw]*len(Vdot))
    
    
    ##### FIGURE 2    
    fig, ax = plt.subplots()
    ax.scatter(Vdot, MeanCe,  label='Raw mean data')
    ax.plot(VdotSteps, nlr_model(VdotSteps, *meanNlrParams), color='r',label='Non linear regression curve')
    ax.set_title('Plot of expired NO concentration against expiratory flow rate \n')
    ax.set_xlabel('Flow (mL/s)')
    ax.set_ylabel('Expired NO concentration (ppb)')
    ax.set_xlim(-50, 2000)
    ax.set_ylim(0, 150)
    ax.grid()
    ax.legend()
    ax.annotate(' Cw = %.2f ppb \n Dno = %.2f nL/s/ppb x 10^-3 \n Calv =  %.2f ppb'
                %(meanNlrParams[0], meanNlrParams[1], meanNlrParams[2]),
                xy=(200, 160), xycoords='figure pixels')
    fig.savefig('Figure2.png')
    plt.show()
    
    ##### FIGURE 3A
    
    fig, ax = plt.subplots()
    ax.scatter(Vdot, qdot,color='black', label = 'qdot raw data')
    ax.plot(VdotSteps, nlr_model(VdotSteps,*meanNlrParams) * VdotSteps, color='r', label = 'qdot nonlinear regression')
    ax.plot(VdotSteps, nlr_model(VdotSteps,*meanNlrParams) * VdotSteps - (VdotSteps*Calv), label = 'qdot_D')
    plt.plot(Vdot, DnoCw, '--')
    ax.grid()
    plt.ylim(0,7000)
    plt.xlim(-50,1000)
    ax.set_title('Plot of total NO output (qdot) and output from airway NO diffusion(qdot_D) \nalone against expiratory flow rate (0 to 1L/s)')
    ax.set_xlabel('Flow (mL/s)')
    ax.set_ylabel('Expired NO output (mL/s)')
    ax.legend()
    ax.annotate('Dno*Cw', xy=(330, 70), xycoords='figure pixels')
    fig.savefig('Figure3A.png')
    plt.show()


    ########## FIGURE 3B
    
    fig, ax = plt.subplots()
    ax.scatter(Vdot, qdot, label = 'qdot, raw data')
    ax.plot(VdotSteps, nlr_model(VdotSteps, *meanNlrParams) * VdotSteps, color='r', label = 'qdot, nonlinear regression')
    ax.plot(VdotSteps, nlr_model(VdotSteps,*meanNlrParams) * VdotSteps - (VdotSteps*Calv), label = 'qdot_D')
    plt.plot(Vdot, DnoCw, '--')
    ax.grid()
    plt.ylim(0,1000)
    plt.xlim(0,20)
    ax.set_title('Plot of total NO output (qdot) and output from airway NO diffusion(qdot_D) \nalone against expiratory flow rate (0 to 20mL/s)')
    ax.set_xlabel('Flow (mL/s)')
    ax.set_ylabel('Expired NO output (nL/s)')
    ax.legend()
    fig.savefig('Figure3B.png')
    plt.show()
    






def points6_linear_regression():
    
    print('*'*71)
    print('*'*22, '6 point linear regression', '*'*22)
    print('*'*71)
    
    ##### equation: qdot = -Dno*Ce + Dno*Cw
    
    Dno6points = []
    Cw6points = []
    Corr6points = []
    for patient in range(10):
        reg_lin =  LinearRegression().fit(PatientsCe[patient][:6][:,np.newaxis],
                                         PatientsCe[patient][:6]*Vdot[:6])
        Dno6points.append(abs(reg_lin.coef_[0]))
        Cw = -reg_lin.intercept_ / reg_lin.coef_[0] #qdot = intercept_y and Ce=0
        Cw6points.append(Cw)

        corrScore = np.corrcoef(np.array(PatientsCe[patient][:6], dtype= np.float32),
                                np.array(PatientsCe[patient][:6]*Vdot[:6], dtype= np.float32))
        Corr6points.append(abs(corrScore[0,1]))
    
    
    lrParams = []
    lrParams.append(list(np.array(Dno6points).round(1)))
    lrParams.append(list(np.array(Cw6points).round(1)))
    lrParams.append(list(np.array(Corr6points).round(3)))
    lrParams = np.swapaxes(np.array(lrParams), axis1=0, axis2=1)
    
    print(pd.DataFrame(lrParams,
                        index = ['patient %d'%x for x in range(1,11)],
                        columns = ['Dno (nL/s/ppb x 10^-3)', 'Cw (ppb)', 'r']))    
    
    lrParams = []
    lrParams.append(Dno6points)
    lrParams.append(Cw6points)
    lrParams.append(Corr6points)
    lrParams = np.swapaxes(np.array(lrParams), axis1=0, axis2=1)
    # lrParams = np.delete(lrParams,5, axis=0) # not remove patient 6


    lrParamsGeoMean = np.exp(np.sum(np.log(lrParams), axis=0)/(len(lrParams))) # careful here divide by 9 instead of 10
    lrParamsSem = lrParamsGeoMean * np.std(np.log(lrParams), axis=0)/np.sqrt(len(lrParams))     # here we use 10 as n but in the paper they used n=9 error?

    print('-'*30)
    print('Mean values for 6point linear: \nDno = %.2f nL/s/ppb x 10^-3 \nCw =  %.2f ppb'%(lrParamsGeoMean[0],
                                                                         lrParamsGeoMean[1]))
    print('-'*30)
    print('SEM values for 6point linear: \nDno = %.2f nL/s/ppb x 10^-3 \nCw =  %.2f ppb'%(lrParamsSem[0],
                                                                        lrParamsSem[1]))
    print('-'*30)
    
    
    ##### FIGURE 4

    # raw data
    Ce = MeanCe[:6]
    qdotmean = Ce*Vdot[:6]
    
    #linear regression
    reg_mean = LinearRegression().fit(Ce[:,np.newaxis], qdotmean)
    
    #non_linear regression
    meanNlrParams, meanNlrCov = curve_fit(f = non_linear_regression_Ce,
                                          xdata = Vdot,
                                          ydata = MeanCe)
    qdotSteps = nlr_model(VdotSteps, *meanNlrParams) * VdotSteps
    CeSteps = nlr_model(VdotSteps, *meanNlrParams)
    qdotSteps = np.insert(qdotSteps, 0, 0)
    CeSteps = np.insert(CeSteps, 0, meanNlrParams[0])
    
    
    fig, ax = plt.subplots()
    ax.scatter(Ce, qdotmean*10**-3, color='black',label='raw data 6')
    ax.plot( np.arange(0,200), ((reg_mean.coef_[0]*np.arange(0,200)) + reg_mean.intercept_)*10**-3,
            '--',label='linear regression 6 points' )
    ax.plot(CeSteps, qdotSteps*10**-3, 'r', label = 'qdot nonlinear regression')
    ax.set_title('Plot of qdot and qdot_D against expired NO concentration (Ce)\n 6Points linear regression')
    ax.set_xlabel('Ce (ppb)')
    ax.set_ylabel('qdot (nl/s)')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 3.5)
    ax.axvline(x=meanNlrParams[2], color='k', linestyle='--')
    ax.annotate('Calv', xy=(56, 45), xycoords='figure pixels')
    ax.grid()
    ax.legend()
    fig.savefig('Figure4.png')
    plt.show()




def points2_linear_regression():
    
    ##### equation: qdot = -Dno*Ce + Dno*Cw
    
    print('*'*71)
    print('*'*22, '2 point linear regression', '*'*22)
    print('*'*71)
    
    Dno2points = []
    Cw2points = []
    Corr2points = []
    for patient in range(10):
        reg_lin =  LinearRegression().fit(PatientsCe[patient][[3,5]][:,np.newaxis],
                                         PatientsCe[patient][[3,5]]*Vdot[[3,5]])
        Dno2points.append(abs(reg_lin.coef_[0]))
        Cw = -reg_lin.intercept_ / reg_lin.coef_[0] #qdot = intercept_y and Ce=0
        Cw2points.append(Cw)

        corrScore = np.corrcoef(np.array(PatientsCe[patient][[3,5]], dtype= np.float32),
                                np.array(PatientsCe[patient][[3,5]]*Vdot[[3,5]], dtype= np.float32))
        Corr2points.append(abs(corrScore[0,1]))
    
    
    lrParams = []
    lrParams.append(list(np.array(Dno2points).round(1)))
    lrParams.append(list(np.array(Cw2points).round(1)))
    lrParams.append(list(np.array(Corr2points).round(3)))
    lrParams = np.swapaxes(np.array(lrParams), axis1=0, axis2=1)
    
    print(pd.DataFrame(lrParams,
                        index = ['patient %d'%x for x in range(1,11)],
                        columns = ['Dno (nL/s/ppb x 10^-3)', 'Cw (ppb)', 'r']))    
    
    lrParams = []
    lrParams.append(Dno2points)
    lrParams.append(Cw2points)
    lrParams.append(Corr2points)
    lrParams = np.swapaxes(np.array(lrParams), axis1=0, axis2=1)
    # lrParams = np.delete(lrParams,5, axis=0) # not remove patient 6


    lrParamsGeoMean = np.exp(np.sum(np.log(lrParams), axis=0)/(len(lrParams))) # careful here divide by 9 instead of 10
    lrParamsSem = lrParamsGeoMean * np.std(np.log(lrParams), axis=0)/np.sqrt(len(lrParams))     # here we use 10 as n but in the paper they used n=9 error?

    print('-'*30)
    print('Mean values for 2point linear: \nDno = %.2f nL/s/ppb x 10^-3 \nCw =  %.2f ppb'%(lrParamsGeoMean[0],
                                                                         lrParamsGeoMean[1]))
    print('-'*30)
    print('SEM values for 2point linear: \nDno = %.2f nL/s/ppb x 10^-3 \nCw =  %.2f ppb'%(lrParamsSem[0],
                                                                        lrParamsSem[1]))
    print('-'*30)
    
    
    ##### FIGURE 4bis

    # raw data
    Ce = MeanCe[[3,5]]
    qdotmean = Ce*Vdot[[3,5]]
    
    #linear regression
    reg_mean = LinearRegression().fit(Ce[:,np.newaxis], qdotmean)
    
    #non_linear regression
    meanNlrParams, meanNlrCov = curve_fit(f = non_linear_regression_Ce,
                                          xdata = Vdot,
                                          ydata = MeanCe)
    qdotSteps = nlr_model(VdotSteps, *meanNlrParams) * VdotSteps
    CeSteps = nlr_model(VdotSteps, *meanNlrParams)
    qdotSteps = np.insert(qdotSteps, 0, 0)
    CeSteps = np.insert(CeSteps, 0, meanNlrParams[0])
    
    
    fig, ax = plt.subplots()
    ax.scatter(Ce, qdotmean*10**-3, color='black',label='raw data 6')
    ax.plot( np.arange(0,200), ((reg_mean.coef_[0]*np.arange(0,200)) + reg_mean.intercept_)*10**-3,
            '--',label='linear regression 6 points' )
    ax.plot(CeSteps, qdotSteps*10**-3, 'r', label = 'qdot nonlinear regression')
    ax.set_title('Plot of qdot and qdot_D against expired NO concentration (Ce)\n 2Points linear regression')
    ax.set_xlabel('Ce (ppb)')
    ax.set_ylabel('qdot (nl/s)')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 3.5)
    ax.axvline(x=meanNlrParams[2], color='k', linestyle='--')
    ax.annotate('Calv', xy=(56, 45), xycoords='figure pixels')
    ax.grid()
    ax.legend()
    fig.savefig('Figure4bis.png')
    plt.show()
    
    
    
    

if __name__ == '__main__':

    tic = time.perf_counter()

    
    #############################################
    #######       VARIABLE INIT          ########
    #############################################

    print('-'*30)
    print('Loading data from feno_subjects.xlsx...')
    print('Variables init...')
    print('-'*30)
    
    df = pd.read_excel('feno_subjects.xlsx',header = None) 
    df = np.array(df) 

    Vdot = df[0,1:]  # 9 flow rates in mL/s
    VdotSteps = np.arange(1, 2000, 0.1) # used to plot curves
    PatientsCe = df[1:11,1:]  # 10 patients Ce values x 9 flow rates in ppb
    MeanCe = df[11,1:] # mean values of Ce per flow rate in ppb
    SemCe = df[12,1:] # sem values of Ce per flow rate in ppb
    
    
    
    #############################################
    ####### 9 POINT NON LINEAR REGRESSION #######
    #############################################
    
    points9_non_linear_regression()    
    
    
    #############################################
    #######   6point linear regression   ########
    #############################################
    
    points6_linear_regression()
    
    
    #############################################
    #######   2point linear regression   ########
    #############################################
    
    points2_linear_regression()
    
    
    
    ########################################
    #############    TIMER     #############
    ########################################
    
    toc = time.perf_counter()
    print('\n toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
      
