import os
import sys
import pickle
import warnings
import numpy as np 
import pandas as pd 
from ast import literal_eval
import matplotlib.pyplot as plt
from scipy import sparse, stats
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import train_test_split
import argparse 
import math 
import warnings

import ppca
import hpmf
import baseline
import pca
import DEF


def get_ratings_matrix(df, train_size=0.75):
    user_to_row = {}
    movie_to_column = {}
    df_values = df.values
    n_dims = 10
    parameters = {}
    
    uniq_users = np.unique(df_values[:, 0])
    uniq_movies = np.unique(df_values[:, 1])

    for i, UserId in enumerate(uniq_users):
        user_to_row[UserId] = i

    for j, ItemId in enumerate(uniq_movies):
        movie_to_column[ItemId] = j
    
    n_users = len(uniq_users)
    n_movies = len(uniq_movies)
    
    R = np.zeros((n_users, n_movies))
    
    df_copy = df.copy()
    train_set = df_copy.sample(frac=train_size, random_state=0)
    test_set = df_copy.drop(train_set.index)
    
    for index, row in train_set.iterrows():
        i = user_to_row[row.userId]
        j = movie_to_column[row.movieId]
        R[i, j] = row.rating

    return R, train_set, test_set, n_dims, n_users, n_movies, user_to_row, movie_to_column

def matrix_X(R):
  X = []
  for i in range(len(R)):
    row = [1 if val == 1 else 0 for val in R[i]]
    X.append(row)
  return X


def recommender(X, y, pmfU):
    y_scaler = preprocessing.StandardScaler().fit(y)
    y_scaled = y_scaler.fit_transform(y)

    X_scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = X_scaler.fit_transform(X)

    pmfU_scaler = preprocessing.StandardScaler().fit(pmfU)
    pmfU_scaled = pmfU_scaler.fit_transform(pmfU)

    X_train, X_test = train_test_split(X_scaled, test_size=0.20, random_state=randseed)
    y_train, y_test = train_test_split(y_scaled, test_size=0.20, random_state=randseed)
    pmfU_train, pmfU_test = train_test_split(pmfU_scaled, test_size=0.20, random_state=randseed)
    n_users, n_items = X_train.shape

    reg = linear_model.Ridge(normalize=True)
    for i in range(n_items):
        # if i%100 == 0:
        #   print('---- Fitting row', i, '----')
        reg.fit(np.column_stack([X_train[:,i], pmfU_train]), y_train[:,i])

    #estimate potential ratings
    test_items = X_test.shape[1]
    prediction = []

    for i in range(test_items):
        # if i%100 == 0:
        #   print('---- Predicting row', i, '----')
        res = reg.predict(np.column_stack([X_test[:,i], pmfU_test]))
        prediction.append(res)

    #evaluate model
    y_test = np.transpose(y_test)
    rmse = mean_squared_error(y_test, prediction, squared=False)
    print("RSME: {}",rmse)



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()

    parser.add_argument('--df', help='dataframe')
    parser.add_argument('--factorModel', help='type of factor model, select from baseline, hpmf, ppca_l, ppca_q, pca, def')
    parser.add_argument('--latentDim', help="Number of latent Dimensions")
    parser.add_argument('--epochs', help="Number of training epochs")
    args = parser.parse_args()

    df = args.df #preprocess this according to your data, item (movie, book, etc) must have itemID as key"

    #get ratings matrix
    R, train_set, test_set, n_dims, n_users, n_movies, user_to_row, movie_to_column = get_ratings_matrix(df, 0.8)

    if (args.factorModel=="baseline"):
        
        Baseline = baseline.baseline(R, train_set, test_set, n_dims, n_users, n_movies, user_to_row, movie_to_column,args.epochs)
        log_ps, rmse_train, rmse_test, p = Baseline.train()

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        plt.title('Training results')
        ax1.plot(np.arange(len(log_ps)), log_ps, label='MAP')
        ax1.legend()

        ax2.plot(np.arange(len(rmse_train)), rmse_train, label='RMSE train')
        ax2.plot(np.arange(len(rmse_test)), rmse_test, label='RMSE test')
        ax2.legend()

        plt.show()
        print('RMSE of training set:', evaluate(train_set))
        print('RMSE of testing set:', evaluate(test_set))

        return

    if (args.factorModel=="hpmf"):
        param, recommender, exposure_df, train, test, n_users, n_movies, user_to_row, movie_to_column = hpmf(df)
        #latent factor matrix U for recommender, based on the factor model used
        FactorU = param.Theta

    elif (args.factorModel=="ppca_l"):
        PPCA = ppca(df, args.latentDim, "linear")
        z_mean_inferred, z_stddv_inferred= PPCA.GetRowFactors()
        #latent dim for recommender
        factorU = z_mean_inferred

    elif (args.factorModel=="ppca_q"):
        PPCA = ppca(df, args.latentDim, "quadratic")
        z_mean_inferred, z_stddv_inferred= PPCA.GetRowFactors()
        #latent dim for recommender
        factorU = z_mean_inferred

    elif(args.factorModel=="ppca_q"):
        p = PCA(df, args.latent_dim)
        userFactors, itemFactors = p.GetFactorsForRowsAndColumns()
        factorU = userFactors

    elif(args.factorModel=="def"):
        z_b = DEF.DEF_main(df, args.epochs)
        factorU = z_b.detach().numpy()


    #recommendation
    X = matrix_X(R)
    ratings = df['rating']
    y = R
    recommender(X, y, factorU)

   







