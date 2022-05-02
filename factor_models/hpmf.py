
import pandas as pd, numpy as np
from hpfrec import HPF


def exposure_data(df, , train_size=0.75):
    user_to_row = {}
    movie_to_column = {}
    
    uniq_users = np.unique(df['userId'])
    uniq_movies = np.unique(df['itemID'])

    for i, user_id in enumerate(uniq_users):
        user_to_row[user_id] = i

    for j, movie_id in enumerate(uniq_movies):
        movie_to_column[movie_id] = j
    
    data = []
    n_users = len(uniq_users)
    n_movies = len(uniq_movies)
    for row in df.iterrows():
      user, movie = row[1][0], row[1][1]
      data.append((user_to_row[user], movie_to_column[movie], 1))

    exposure_df = pd.DataFrame(data, columns =['userID', 'itemID', 'a'])
    #Train-test split
    df_copy = exposure_df.copy()
    train_set = df_copy.sample(frac=train_size, random_state=0)
    test_set = df_copy.drop(train_set.index)
    
    return exposure_df, train_set, test_set, n_users, n_movies, user_to_row, movie_to_column

def matrix_X(R):
  X = []
  for i in range(len(R)):
    row = [1 if val == 1 else 0 for val in R[i]]
    X.append(row)
  return X



def hpmf(df):
    exposure_df, train, test, n_users, n_movies, user_to_row, movie_to_column = exposure_data(df)

    exposure_df.columns = ['userId', 'itemId', 'Count']
    recommender = HPF()
    param = recommender.fit(exposure_df)
    return param, recommender, exposure_df, train, test, n_users, n_movies, user_to_row, movie_to_column