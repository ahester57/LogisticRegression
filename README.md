# Logistic Regression  
### Austin Hester
### CS 4340
### Uday Chakraborty

Uses logistic regression to formulate a correlation between weeks 
of inactivity and probability of passing the course.  

Our data:  
X	|	Y  
1	|	0  
2	|	1  
3	|	0  
4	|	1  
5	|	0  
6	|	1  
7	|	1  
8	|	1  

We use logistic to regression to find a weight vector.  
With a step size, c = 0.01, and 2000 iterations,    
we obtain w_v = { -1.81 , 0.55 }.  

We can use these weights to find the chance of passing given
weeks of inactity.

```P ( Y(j) = 0 | X(j) ) = 1 / (1 + e^(-1.81 + (X(j) x 0.55)))```

At x = 3, we get  

```[P ( Y = 0 | x = 3 ) = 1 / (1 + e^(-1.81 + (3 x 0.55))) = 0.535]```


