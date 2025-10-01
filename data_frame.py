import pandas as pd
import numpy as np
from astropy.table import Table
from itertools import combinations
from abc import ABC, abstractmethod

VALUES_TO_FILL = [-99.0, 99.0]
MAGNITUDES = ['MAG_GAAP_u','MAG_GAAP_g', 'MAG_GAAP_r','MAG_GAAP_i1', 'MAG_GAAP_i2',
              'MAG_GAAP_Z', 'MAG_GAAP_Y','MAG_GAAP_J', 'MAG_GAAP_H','MAG_GAAP_Ks']

MAG_ERRS = ['MAGERR_GAAP_u','MAGERR_GAAP_g','MAGERR_GAAP_r','MAGERR_GAAP_i1','MAGERR_GAAP_i2',
            'MAGERR_GAAP_Z','MAGERR_GAAP_Y','MAGERR_GAAP_J', 'MAGERR_GAAP_H','MAGERR_GAAP_Ks']

COLORS = ['u-g', 'u-r', 'u-i1', 'u-i2', 'u-Z', 'u-Y','u-J', 'u-H', 'u-Ks', 'g-r', 'g-i1', 'g-i2',
          'g-Z', 'g-Y', 'g-J', 'g-H','g-Ks', 'r-i1', 'r-i2', 'r-Z', 'r-Y', 'r-J', 'r-H', 'r-Ks', 
          'i1-i2','i1-Z', 'i1-Y', 'i1-J', 'i1-H', 'i1-Ks', 'i2-Z', 'i2-Y', 'i2-J', 'i2-H','i2-Ks', 
          'Z-Y', 'Z-J', 'Z-H', 'Z-Ks', 'Y-J', 'Y-H', 'Y-Ks', 'J-H','J-Ks', 'H-Ks']

class MissingValueFiller(ABC):
    
    @abstractmethod
    def fill(self, data: pd.DataFrame):
        pass

class MeanFiller(MissingValueFiller):

    def fill(self, data):
        return data.fillna(data.loc[~data["faint"]].mean(numeric_only=True))

class MedianFiller(MissingValueFiller):

    def fill(self, data):
        return data.fillna(data.loc[~data["faint"]].median(numeric_only=True))

class MaxFiller(MissingValueFiller):

    def fill(self, data):
        return data.fillna(data.loc[~data["faint"]].max(numeric_only=True))

class MinFiller(MissingValueFiller):

    def fill(self, data):
        return data.fillna(data.loc[~data["faint"]].min(numeric_only=True))

class DataFrame:
    
    def __init__(self, filePath, objtype, filler=None, process=True):
        self.filler = filler
        self.filePath = filePath
        self.objtype = objtype
        self.data = None
        self.features = MAGNITUDES + COLORS

        self.load()
        if process:
            self.data = self.data.replace(VALUES_TO_FILL, np.nan)
            self.data["split"] = "train"
            self.data["faint"] = False
            self.data["has_missing"] = self.data.isna().any(axis=1)
            self.is_faint()
            self.fill()
            self.make_colors()
            self.train_val_test_split()
            self.data["Z_pred"] = 0
            self.data["Z_pred_std"] = 0
        

    def set_filler(self, filler: MissingValueFiller):
         self.filler = filler

    def set_objtype(self, objtype):
        self.objtype = objtype
    
    def load(self):
        self.data = Table.read(self.filePath, format='fits').to_pandas()
        self.data["SPECTYPE"] = self.data["SPECTYPE"].apply(lambda x: x.decode('utf-8').strip())
        self.data = self.data[self.data["SPECTYPE"] == self.objtype]

    def fill(self):
        if self.filler:
            self.data = self.filler.fill(self.data)

    def is_faint(self):
        indexes = self.data.nlargest(int(0.1 * len(self.data)), "MAG_GAAP_r").index
        self.data.loc[indexes, "faint"] = True

    def make_colors(self):
        pairs = list(combinations(MAGNITUDES, 2))
        for pair in pairs:
            mag1 = pair[0]
            mag2 = pair[1]
            
            self.data[f"{mag1.split('_')[-1]}-{mag2.split('_')[-1]}"] = self.data[mag1] - self.data[mag2]

    def train_val_test_split(self):
        rng = np.random.default_rng(seed=42)
        self.data.loc[~self.data["faint"],"split"] = rng.choice(['train', 'val', 'test'], size=len(self.data[~self.data["faint"]]), p=[0.8, 0.1, 0.1])
        self.data.loc[self.data["faint"],"split"] = rng.choice(['val', 'test'], size=len(self.data[self.data["faint"]]), p=[0.5,0.5])

    def get_train_dataset(self):
        return self.data[self.data["split"] == "train"][self.features], self.data[self.data["split"] == "train"]["Z"]

    def get_val_dataset(self):
        return self.data[(self.data["split"] == "val") & (~self.data["faint"])][self.features], self.data[(self.data["split"] == "val") & (~self.data["faint"])]["Z"]

    def get_faint_val_dataset(self):
        return self.data[(self.data["split"] == "val") & (self.data["faint"])][self.features], self.data[(self.data["split"] == "val") & (self.data["faint"])]["Z"]

    def get_test_dataset(self):
        return self.data[self.data["split"] == "test"][self.features], self.data[self.data["split"] == "test"]["Z"]

    def get_random_test_dataset(self): 
        return self.data[(self.data["split"] == "test") & (~self.data["faint"])][self.features], self.data[(self.data["split"] == "test") & (~self.data["faint"])]["Z"]
    
    def get_faint_test_dataset(self):
        return self.data[(self.data["split"] == "test") & (self.data["faint"])][self.features], self.data[(self.data["split"] == "test") & (self.data["faint"])]["Z"]

    def load_df(self, filePath):
        self.data = pd.read_csv(filePath, index_col=0)
        


