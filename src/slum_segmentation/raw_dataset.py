from types import SimpleNamespace

class City:
    def __init__(self, name):
        self.name = name

    def get_image_path(self): return f"../data/raw/{self.name}/image.tif"
    def get_mask_path(self): return f"../data/raw/{self.name}/mask.tif"

cities = SimpleNamespace(
    capetown = City("capetown"),
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