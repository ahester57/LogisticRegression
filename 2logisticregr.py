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
        
        eexp = np.exp( ( w_[0] * x0 ) + ( w_[1] * x[j-1][0] ) +
                            ( w_[2] * x[j-1][1]) ) 
        if (i == 0):
            pointmult = 1
        else:
            pointmult = x[j-1][i-1]
        point = pointmult * (y[j-1] - ( eexp / ( 1 + eexp ) ) ) 
        s = s + point
    w = w_[i] + (c * s)
    return w

# compute the passing chance given x weeks of inactivity
def passingchance(w_, x):
    chance = 1 / ( 1 + np.exp( x0 * w_[0] + (x[0] * w_[1]) + (x[1]*w_[2])))
    return chance

# input data [1-8], "weeks of inactivity" , " avg AP score; 0=N/A"
#x = np.arange(1, 9)
x = np.array( [[1,1,2,3,3,4,4,5,5,6,7,8],
               [0,4,2,0,0,2,4,5,3,3,4,5]])
x = x.T
# output, 0 = "pass", 1 = "fail"
y = np.array(  [0,0,1,0,1,1,0,0,1,1,1,1] )
# step size
c = 0.1
# initial weight vector
w_ = np.array( [1., 1., 1.] )

# iterate T times
T = 2000
for t in range(T):
    print("===========================")
    new_w0 = ddw(0,x,y,w_,c)
    print(new_w0)
    print("---------------------------")
    new_w1 = ddw(1,x,y,w_,c)
    print(new_w1)
    print("---------------------------")
    new_w2 = ddw(2,x,y,w_,c)
    print(new_w2)
    w_[0] = new_w0
    w_[1] = new_w1
    w_[2] = new_w2
    print("===========================")

# print weight vector
print("\nWeight vector: ", w_)
print("\nWeeks of Inactivity")
print("x avg AP exam\t\tChances of passing")
for i in range(0,12):
    print("\t",x[i], "\t\t", round(passingchance(w_, x[i]),4)*100, "%")

