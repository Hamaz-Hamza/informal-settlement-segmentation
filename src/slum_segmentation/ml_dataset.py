from types import SimpleNamespace
import numpy as np

class City:
    def __init__(self, name):
        self.name = name

    def get_train_data(self): return (
        np.load(f"../data/machine_learning_splits/{self.name}/x_train.npy"),
        np.load(f"../data/machine_learning_splits/{self.name}/y_train.npy"),
    )

    def get_val_data(self): return (
        np.load(f"../data/machine_learning_splits/{self.name}/x_val.npy"),
        np.load(f"../data/machine_learning_splits/{self.name}/y_val.npy"),
    )
    
    def get_test_data(self): return (
        np.load(f"../data/machine_learning_splits/{self.name}/x_test.npy"),
        np.load(f"../data/machine_learning_splits/{self.name}/y_test.npy"),
    )

cities = SimpleNamespace(
    combined = City("combined"),
    capetownsmall = City("capetownsmall"),
    eldaein = City("eldaein"),
    elgeneina = City("elgeneina"),
    kianda = City("kianda"),
    kibera = City("kibera"),
    lower = City("lower"),
    makoko = City("makoko"),
    medellin = City("medellin"),
    mumbai = City("mumbai"),
)