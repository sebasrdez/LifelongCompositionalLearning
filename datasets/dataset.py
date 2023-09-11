import pandas as pd
from torch.utils.data import TensorDataset
import torch
import numpy as np

class MyDataset:
    
    def __init__(self, num_tasks, paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, regression=True):

        # Ensure that the sum of data splits is 1
        assert train_ratio + val_ratio + test_ratio == 1., "Los porcentajes de cada subconjunto de datos deben sumar 1."

        self.num_total_tasks = num_tasks
        self.num_tasks = min(num_tasks, self.num_total_tasks)

        self.paths = paths

        self.trainset = []
        self.valset = []
        self.testset = []

        self.features = []

        self.max_batch_size = 0

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.regression = regression

        self.load_data()

    # Load and preprocess data from paths
    def load_data(self):
        
        # Ensure number of data files match number of tasks
        assert len(self.paths)==self.num_tasks, "El número de archivos de datos debe ser igual al número de tareas."

        for path in self.paths:
            df = pd.read_csv(path)

            # One-hot encode 'Accion' column
            df = pd.get_dummies(df, columns=['Accion'])

            # Extract target (output) and features (inputs) from dataframe
            target = df.filter(regex='^(OUT)', axis=1).values
            features = df.filter(regex='^(Accion|IN)', axis=1).values

            # Shuffle indices for creating training, validation, and test splits
            idx_shuffle = np.random.permutation(len(target))
            
            # Calculate split sizes
            num_train = int(len(target) * self.train_ratio)
            num_val = int(len(target) * self.val_ratio)

            # Slice data into respective splits based on shuffled indices
            X_train = features[idx_shuffle[:num_train]]
            y_train = target[idx_shuffle[:num_train]]
            X_val = features[idx_shuffle[num_train:num_train+num_val]]
            y_val = target[idx_shuffle[num_train:num_train+num_val]]
            X_test = features[idx_shuffle[num_train+num_val:]]
            y_test = target[idx_shuffle[num_train+num_val:]]
            
            X_train = X_train.astype('float64')
            X_val = X_val.astype('float64')
            X_test = X_test.astype('float64')
            y_train = y_train.astype('float64')
            y_val = y_val.astype('float64')
            y_test = y_test.astype('float64')

            y_train = torch.from_numpy(y_train).float().unsqueeze(dim=1)
            y_val = torch.from_numpy(y_val).float().unsqueeze(dim=1)
            y_test = torch.from_numpy(y_test).float().unsqueeze(dim=1)

            X_train = torch.from_numpy(X_train).float()
            X_val = torch.from_numpy(X_val).float()
            X_test = torch.from_numpy(X_test).float()

            # Store data splits as tensor datasets for PyTorch
            self.trainset.append(TensorDataset(X_train, y_train))
            self.valset.append(TensorDataset(X_val, y_val))
            self.testset.append(TensorDataset(X_test, y_test))

            self.features.append(X_train.shape[1])
            self.max_batch_size = max(self.max_batch_size, X_train.shape[0], X_val.shape[0], X_test.shape[0])