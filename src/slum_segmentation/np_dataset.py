from types import SimpleNamespace
import numpy as np

class City:
    def __init__(self, name):
        self.name = name

    def get_image(self): return np.load(f"../data/preprocessed/{self.name}/image.npy")
    def get_mask(self): return np.load(f"../data/preprocessed/{self.name}/mask.npy")

cities = SimpleNamespace(
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