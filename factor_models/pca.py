#Fitting exposure matrix using PCA
from sklearn.decomposition import PCA
from sklearn import preprocessing

class PCA():
    def __init__(df, latent_dim):
        self.df = df.copy()
        self.latent_dim = latent_dim
        self.train_size = 0.75
        self.user_to_row = {}
        self.movie_to_column = {}
        
        self.uniq_users = np.unique(df['userId'])
        self.uniq_movies = np.unique(df['itemId'])

        for i, user_id in enumerate(self.uniq_users):
            self.user_to_row[user_id] = i

        for j, movie_id in enumerate(self.uniq_movies):
           self.movie_to_column[movie_id] = j
        
        data = []
        self.n_users = len(self.uniq_users)
        self.n_movies = len(self.uniq_movies)
        for row in self.df.iterrows():
        user, movie = row[1][0], row[1][1]
        data.append((user_to_row[user], movie_to_column[movie], 1))

        self.exposure_df = pd.DataFrame(data, columns =['userid', 'movieid', 'a'])
        #Train-test split
        self.train_set = df_copy.sample(frac=self.train_size, random_state=0)
        self.test_set = df_copy.drop(self.train_set.index)
        

    #Row represent userIds, Columns represent songIds
    #dim are number of latent factors
    #exposureMatrix is numpy matrix, with row represents every user and column representing every feature
    def GetFactorsForRowsAndColumns(self):
        Xscaler = preprocessing.StandardScaler(with_std=False).fit(self.exposureMatrix)
        Xscaled = Xscaler.fit_transform(self.exposureMatrix)
        pca = PCA(n_components=self.latent_dim)
        pca.fit(Xscaled)
        rowFactors = pca.fit_transform(Xscaled)
        colFactors = pca.components_.T

        #Shape of row factors will be (num of rows) x (latent dim)
        #Shape of column factors will be (num of cols) x (latent dim)
        
        return rowFactors, colFactors