import numpy as np
# import pandas as pd


class Guassian:
    def __init__(self,mean,cov_mat,num_c,ranges):
        self.mean = mean
        self.cov_mat = cov_mat
        self.num_c = num_c
        self.x1 = ranges[0]
        self.y1 = ranges[1]

    # Function to Apply normal distribution formula for a particular N dimensional point
    def norm_dis(self,x):
        '''
        INPUTS:
            x is a vector representing the number of features (dimensions) => numpy array
            mean is the mean of N distributions => 1 D (N length) numpy array
            cov_mat is the covariance matrix => N-dimensional numpy array
            num_c is the number of dimensions => integer
        OUTPUT:
            returns the value of normal distribution at given x for the corresponding mean and covariance
        '''
        den = (1/((2*np.pi)**(self.num_c/2))*(np.linalg.det(self.cov_mat)**(0.5)))
        diff_x_mean = np.subtract(x,self.mean)
        exp_val = np.matmul(np.matmul(np.transpose(diff_x_mean),np.linalg.inv(self.cov_mat)),(diff_x_mean))
        return den*(np.exp((-0.5)*exp_val))

    # Function to Iterate over the points to get the overall distribution
    def iter_norm_dis(self):
        '''
        INPUT:
            x1,y1 are 2D values of range whose meshgrid is to be created to evaluate the function
            at multiple points
            => numpy arrays
        OUTPUT:
            returns an N-dimensional (here 2 D mostly) with the values from the norm_dist function
        '''
        [x1,y1] = np.meshgrid(self.x1,self.y1,indexing="ij")
        z = np.zeros((x1.shape[0],y1.shape[1]))
        
        # for each location, find the probability value and store it in z
        for i in range(x1.shape[0]):
            for j in range(y1.shape[1]):
                z[i][j] = self.norm_dis(np.array([x1[i][j],y1[i][j]]))
        return (z,x1,y1)


    def find_dist(self,test_pt):
        diff_x_mean = np.subtract(test_pt,self.mean)
        return np.sqrt(np.matmul(np.matmul(np.transpose(diff_x_mean),np.linalg.inv(self.cov_mat)),(diff_x_mean)))
        # print(np.subtract(self.mean,test_pt))

# Configuring the number of classes (Currently works only for 2D)
# number_of_classes = 2
# mean = np.zeros(number_of_classes)