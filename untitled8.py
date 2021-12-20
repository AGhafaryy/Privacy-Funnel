# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:26:32 2021

@author: 96176
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

PSX = np.array([[0.0625, 0.0625, 0.03125, 0.03125], [0.125, 0.0625, 0.03125, 0.03125], [0.0625, 0.0625, 0.0625, 0.0625],[0.25, 0,  0, 0.0625]])
PX = [0.25,0.25,0.25,0.25] #Check eza l marginal 
PYgX = np.identity(4) #start with P Y given X as an identity matrix

def get_PYX():
    return np.dot(PX,PYgX)

def matrix_multiplication(a,b): 
    c = np.matmul(a,b) #c being the result of the product
    return c

def entropy_term(x): #base e
    if x==0: return 0
    else: return -x*math.log(x)
    
def merge_column( a, i , j ): # merging columns function
   a_del= copy.deepcopy(a)
   for x in range (len(a)):
       a_del[x][i] = (a_del[x][i]+ a_del[x][j])
   a_del = np.delete(a_del, j, 1)
   return a_del

def entropy_row(a): #this finds the marginal entropy of the element that is on the rows, given a joint distribution 
    sum = 0
    sum_f = 0
    for i in range (len(a)):
        for j in range (len(a[0])):
            sum += a[i][j]
        sum_f += entropy_term(sum)
        sum = 0
    return sum_f

def entropy_column(a): #this finds the marginal entropy of the element that is on the columns , given a joint distribution
    sum = 0
    sum_f = 0
    for i in range (len(a[0])):
        for j in range (len(a)):
            sum += a[j][i]
        sum_f += entropy_term(sum)
        sum = 0
    return sum_f 

def joint_entropy(a):
   sum = 0
   for i in range (len(a)):
       for j in range(len(a[0])):
           sum += entropy_term(a[i][j])
   return sum

def HrowGivencolumn(a): #this function gives you conditional entropy of element on rows given that on the columns
    return joint_entropy(a)-entropy_column(a) 

def mutual_info(a):
    return entropy_row(a)-HrowGivencolumn(a)

def get_PSY(a,b): # a being PYgX , b is PSX: X rows, S columns 
    PYgXtrans = np.transpose(a)
    PSY = np.dot(PYgXtrans,b)
    return np.transpose(PSY) 
#Y aal rows, S columns

#Since I have PSXY: 
#PSXY = P(Y)*PXgY*PSgX = PY*PXgY*PSgX
#PSY = PY*PSgY

b = [[0],[0],[0],[0]]
def get_PY_S(a): #a bieng PSY  marginal taba3 l Y mnl PSY
    for i in range(len(b)):
        sum = 0
        for j in range(4): 
            sum = sum + get_PSY(PYgX,PSX)[i][j]
        b[i][0]=sum
    return b    

def get_PSgY(a,b): # a being PYgX, b is PSY
    return np.dot(np.transpose(get_PY_S(b)),get_PSY(a,PSX))


z=mutual_info(get_PSY(PYgX,PSX))

# =============================================================================
def privacy_funnel(PYgX,PSX,R,min):
     for k in range(0,2):
         for i in range(len(PYgX[0])):
             for j in range(i+1,len(PYgX[0])):
                 x=merge_column(PYgX,i,j)
                 if(mutual_info(np.dot(np.diag(PX),x))>=R): 
                     if(mutual_info(get_PSY(x,PSX))<min):
                         min = mutual_info(get_PSY(x,PSX))
         PYgX=x
     return min
# =============================================================================
    
# =============================================================================
print(privacy_funnel(PYgX,PSX,0,100))
x = np.linspace(0,1,11)
y = np.linspace(0,1,11)
for i in range (11):
    y[i] = privacy_funnel(PYgX,PSX,x[i],100)
plt.plot(x,y)
#=============================================================================

#print(get_PSgY(PYgX))

#example for i = 0 w j = 1 , c being  I(S; Y ) − I(S; Y i-j)

# =============================================================================
# def PF_function(PYgX,PSX,R,min):
#     for i in range(len(PYgX[0])):
#         for j in range(i+1,len(PYgX[0])):
#             x=merge_column(PYgX,i,j)
#             if(mutual_info(np.dot(np.diag(PX),x))>=R): 
#                 a = get_PYX()[i]+get_PYX()[j]
#                 b = get_PYX()[i]*(get_PSgY(x,get_PSY(x,PSX)))[j]
#                 c = get_PYX()[j]*(get_PSgY(x,get_PSY(x,PSX)))[i]
#                 d = b + c 
#                 if((a*entropy_term(d/a)-d)<min):
#                     min = (a*entropy_term(b/a)-b)
#     return min
#     
# 
# x = np.linspace(0,1,11)
# y = np.linspace(0,1,11)
# for i in range (11):
#     y[i] = PF_function(PYgX,PSX,x[i],100)
# plt.plot(x,y)
# =============================================================================

# =============================================================================
# 
#               if(((get_PYX()[i]+get_PYX()[j])*(entropy_term(get_PYX()[i]*(get_PSgY(PYgX,get_PSY(PYgX,PSX))[j]+get_PYX()[j]*(get_PSgY(PYgX,get_PSY(PYgX,PSX))[i]))/(get_PYX()[i]+get_PYX()[j])))-(get_PYX()[i]*(get_PSgY(PYgX,get_PSY(PYgX,PSX))[j])+get_PYX()[j]*(get_PSgY(PYgX,get_PSY(PYgX,PSX))[i]))) <min):
#                     min = (((get_PYX()[i]+get_PYX()[j])*(entropy_term(get_PYX()[i]*((get_PSgY(PYgX,get_PSY(PYgX,PSX))[j])+get_PYX()[j]*(get_PSgY(PYgX,get_PSY(PYgX,PSX))[i]))/(get_PYX()[i]+get_PYX()[j])))-(get_PYX()[i]*(get_PSgY(PYgX,get_PSY(PYgX,PSX))[j])+get_PYX()[j]*(get_PSgY(PYgX,get_PSY(PYgX,PSX))[i])))
# =============================================================================
