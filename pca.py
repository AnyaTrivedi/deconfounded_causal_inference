#Fitting exposure matrix using PCA
from sklearn.decomposition import PCA
from sklearn import preprocessing






#Row represent userIds, Columns represent songIds
#dim are number of latent factors
#exposureMatrix is numpy matrix, with row represents every user and column representing every feature
def GetFactorsForRowsAndColumns(dim, exposureMatrix):
    Xscaler = preprocessing.StandardScaler(with_std=False).fit(exposureMatrix)
    Xscaled = Xscaler.fit_transform(exposureMatrix)
    pca = PCA(n_components=dim)
    pca.fit(Xscaled)
    rowFactors = pca.fit_transform(Xscaled)
    colFactors = pca.components_.T

    #Shape of row factors will be (num of rows) x (latent dim)
    #Shape of column factors will be (num of cols) x (latent dim)
    
    return rowFactors, colFactors


