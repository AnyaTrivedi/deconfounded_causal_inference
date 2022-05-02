class Baseline():
    def __init__(self, R, train_set, test_set, n_dims, n_users, n_movies, user_to_row, movie_to_column, epochs):
        self.R = R
        self.train_set =  train_set
        self.test_set =  test_set
        self.n_dims = n_dims
        self.n_users = n_users
        self.n_movies = n_movies
        self.user_to_row = user_to_row
        self.movie_to_column = movie_to_column 
        self.epochs = epochs
        
        self.parameters = {}




def initialize_parameters(self, lambda_U, lambda_V):
    U = np.zeros((self.n_dims, self.n_users), dtype=np.float64)
    V = np.random.normal(0.0, 1.0 / lambda_V, (self.n_dims, self.n_movies))
    
    self.parameters['U'] = U
    self.parameters['V'] = V
    self.parameters['lambda_U'] = lambda_U
    self.parameters['lambda_V'] = lambda_V
    
def update_parameters(self):
    U = parameters['U']
    V = parameters['V']
    lambda_U = parameters['lambda_U']
    lambda_V = parameters['lambda_V']
    
    for i in range(self.n_users):
        V_j = V[:, self.R[i, :] > 0]
        U[:, i] = np.dot(np.linalg.inv(np.dot(V_j, V_j.T) + lambda_U * np.identity(self.n_dims)), np.dot(self.R[i, self.R[i, :] > 0], V_j.T))
        
    for j in range(self.n_movies):
        U_i = U[:, self.R[:, j] > 0]
        V[:, j] = np.dot(np.linalg.inv(np.dot(U_i, U_i.T) + lambda_V * np.identity(self.n_dims)), np.dot(self.R[self.R[:, j] > 0, j], U_i.T))
        
    self.parameters['U'] = U
    self.parameters['V'] = V
    
def log_a_posteriori(self):
    lambda_U = parameters['lambda_U']
    lambda_V = parameters['lambda_V']
    U = self.parameters['U']
    V = self.parameters['V']
    
    UV = np.dot(U.T, V)
    R_UV = (self.R[self.R > 0] - UV[self.R > 0])
    
    return -0.5 * (np.sum(np.dot(R_UV, R_UV.T)) + lambda_U * np.sum(np.dot(U, U.T)) + lambda_V * np.sum(np.dot(V, V.T)))   

def predict(self, user_id, movie_id):
    U = self.parameters['U']
    V = self.parameters['V']
    
    r_ij = U[:, self.user_to_row[user_id]].T.reshape(1, -1) @ V[:, self.movie_to_column[movie_id]].reshape(-1, 1)

    max_rating = self.parameters['max_rating']
    min_rating = self.parameters['min_rating']

    return 0 if max_rating == min_rating else ((r_ij[0][0] - min_rating) / (max_rating - min_rating)) * 5.0

def evaluate(self,dataset):
    ground_truths = []
    predictions = []
    
    for index, row in dataset.iterrows():
        ground_truths.append(row.loc['rating'])
        predictions.append(predict(row.loc['userId'], row.loc["itemID"]))
    
    return mean_squared_error(ground_truths, predictions, squared=False)


def update_max_min_ratings(self):
    U = self.parameters['U']
    V = self.parameters['V']

    self.R = U.T @ V
    min_rating = np.min(self.R)
    max_rating = np.max(self.R)

    self.parameters['min_rating'] = min_rating
    self.parameters['max_rating'] = max_rating
    
def train(self):
    self.initialize_parameters(0.3, 0.3)
    log_aps = []
    rmse_train = []
    rmse_test = []

    self.update_max_min_ratings()
    rmse_train.append(evaluate(train_set))
    rmse_test.append(evaluate(test_set))
    
    for k in range(self.n_epochs):
        self.update_parameters()
        log_ap = self.log_a_posteriori()
        log_aps.append(log_ap)

        if (k + 1) % 10 == 0:
            self.update_max_min_ratings()

            rmse_train.append(self.evaluate(train_set))
            rmse_test.append(self.evaluate(test_set))
            print('Log p a-posteriori at iteration', k + 1, ':', log_ap)

    self.update_max_min_ratings()

    return log_aps, rmse_train, rmse_test