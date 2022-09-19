import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy
import random
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

PSX = np.array([[0.0625,0.03125 , 0.0625, 0.03125], [0.125, 0.0625, 0.03125, 0.03125], [0.0625, 0.0625, 0.0625, 0.0625],[0.25, 0,  0, 0.0625]])
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


def get_PY_S(a): #a bieng PSY  marginal taba3 l Y mnl PSY
    b = [0]*len(a)
    for i in range(len(b)):
        sum = 0
        for j in range(4): 
            sum = sum + get_PSY(PYgX,PSX)[i][j]
        b[i]=sum
    return b    

def get_PSgY(a,b): # a being PYgX, b is PSY
    return np.dot(np.transpose(get_PY_S(b)),get_PSY(a,PSX))


z=mutual_info(get_PSY(PYgX,PSX))

def prop_PrivacyFunnel(PYgX,PSX,R,min,h,g):
    for k in range(0,2):
        for i in range(h):
            for j in range(i+1,g):
                x=merge_column(PYgX,i,j)
                print(x)
                if(mutual_info(np.dot(np.diag(PX),x))>=R): 
                    f = np.dot(PX,x)
                    print("Andrew", f)
                    a = f[i]+f[j]
                    print("a", a)
                    b = f[i]*(get_PSgY(x,get_PSY(x,PSX))).item(j)
                    print("b", b)
                    c = f[j]*(get_PSgY(x,get_PSY(x,PSX))).item(i)
                    print("c", c)
                    d = b + c 
                    print("d", d)
                    if((a*entropy_term(d/a)-d)<min):
                        min = (a*entropy_term(d/a)-d)
                        print("min", min)
            g = g-1
        h = h-1
    return min

def privacy_funnel(PYgX,PSX,R,min):
     PYgX_M = PYgX
     for k in range(0,2):
         for i in range(len(PYgX[0])):
             for j in range(i+1,len(PYgX[0])):
                 x=merge_column(PYgX,i,j)
                 if(mutual_info(np.dot(np.diag(PX),x))>=R): 
                     if(mutual_info(get_PSY(x,PSX))<min):
                         min = mutual_info(get_PSY(x,PSX))
                         PYgX_M=x
                 
         PYgX = PYgX_M
         #print("win", PYgX)
     return min
 
l = np.linspace(0,1.37,100)
y = np.linspace(0,1.37,100)
for z in range (100):
    y[z] = privacy_funnel(PYgX,PSX,l[z],0.185)
plt.plot(l,y,"b")


def binary_entropy(a):
    return -a*(math.log(a))-(1-a)*(math.log(1-a))

def star(a,b):
    if(a>=0 and a<=1 and b>=0 and b<=1):
        c = a*(1-b)+b*(1-a)
    return c

r = np.arange(0.0, 10.0 , 1.0)
pp = np.arange(0.0, 10.0 , 1.0)

def GsLemma(i,j):
    p = 0.14
    d = 0.76
    a = 0
    R = binary_entropy(p)-a*binary_entropy(p/(max(a,2*p)))
    PFr = binary_entropy(star(p,d))-a*binary_entropy(star(d,(p/(max(a,2*p)))))-(1-a)*binary_entropy(d)
    r[i] = R
    pp[j] = PFr
    
    
for i in range(10):
    GsLemma(i,i)
#plt.plot(r,pp,"r")
    
def merge(list1, list2):
      
    merged_list = list(zip(list1, list2)) 
    return merged_list
      
T_uple = (merge(r,pp))
T_uple.sort()

plt.plot([ x[0] for x in T_uple], [ x[1] for x in T_uple])

def leakage(PX,a): #sum over Y, if leakage>R
    sum = 0 
    for i in range (3):
        max = 0
        for j in range (len(a[0])):
            if a[j][i]>max:
                max = a[j][i]
        sum = sum + max
        print(sum)
    return math.log(sum)

def max_leakage_function(PYgX,R,min):
     PYgX_M = PYgX
     for k in range(0,2):
         for i in range(len(PYgX[0])):
             for j in range(i+1,len(PYgX[0])):
                 x=merge_column(PYgX,i,j)
                 if(mutual_info(np.dot(np.diag(PX),x))>=R): 
                     if(leakage(PX,x)<min):
                         min = leakage(PX,x)
                         print(min)
                         PYgX_M=x
                 
         PYgX = PYgX_M
         #print("win", PYgX)
     return min


l = np.linspace(0,1.37,100)
y = np.linspace(0,1.37,100)
for z in range (100):
    y[z] = max_leakage_function(PYgX,l[z],0.19)
plt.plot(l,y,"b")
        

            
            