# Austin Hester
# Logistic Regression
# CS 4340 - Intro to Machine Learning
# 11.05.17

import numpy as np
# define x_0
x0 = 1

# ln L = sum_1_n{ x_i^j * ( y^j - ( e^{w0x0+w1x1} / (1 + e^{w0x0+w1x1} )))}

# compute d/dwi ln L with given x, y, weights, and step size
def ddw(i, x, y, w_, c):
    s = 0
    # for d/dw1 ln L
    for j in range(1,9):
        
        eexp = np.exp( ( w_[0] * x0 ) + ( w_[1] * x[j-1] ) ) 
        if (i == 0):
            pointmult = 1
        else:
            pointmult = x[j-1]
        point = pointmult * (y[j-1] - ( eexp / ( 1 + eexp ) ) ) 
        s = s + point
    w = w_[i] + (c * s)
    return w

# compute the passing chance given x weeks of inactivity
def passingchance(w_, x):
    chance = 1 / ( 1 + np.exp(x0 * w_[0] + (x * w_[1])))
    return chance

# input data [1-8], "weeks of inactivity"
#x = np.arange(1, 9)
x = np.array( [1,2,3,4,5,6,7,8] )
# output, 0 = "pass", 1 = "fail"
y = np.array( [0,1,0,1,0,1,1,1] )
# step size
c = 0.01
# initial weight vector
w_ = np.array( [1., 1.] )

# iterate T times
T = 2000
for t in range(T):
    print("===========================")
    new_w0 = ddw(0,x,y,w_,c)
    print(new_w0)
    print("---------------------------")
    new_w1 = ddw(1,x,y,w_,c)
    print(new_w1)
    w_[0] = new_w0
    w_[1] = new_w1
    print("===========================")

# print weight vector
print("\nWeight vector: ", w_)
print("\nWeeks of Inactivity\tChances of passing")
for i in range(0,13):
    print("\t",i, "\t\t", round(passingchance(w_, i),4)*100, "%")

