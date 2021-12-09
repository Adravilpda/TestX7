# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
conway.py 
A simple Python/matplotlib implementation of Conway's Game of Life.
Author: Mahesh Venkitachalam

"""
from sklearn import datasets
import pandas as pn
import numpy as np
from sklearn.neighbors import NearestNeighbors

A=pn.read_csv('cann.txt',skiprows=0,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12])
V1=np.array(A)

B=pn.read_csv('cann.txt',skiprows=0,usecols=[0])
V2=np.array(B)
lineax=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

    
X=V1
#y=V2
print("Y")
print(lineax)
clasificador = NearestNeighbors(n_neighbors=5)
clasificador.fit(X,lineax)
y=clasificador.kneighbors(lineax, return_distance=False)
print(y)
