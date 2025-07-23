import pandas as pd
import glob
import os
import re

class CompareModels:
    def __init__(self, directoryPath):
        self.directoryPath = directoryPath
        self.filePaths = None
        self.modelNames = []
        self.df = None
        self.results = None

        self.loadFiles()
        self.getModelNames()
        self.makeDataFrame()

    def loadFiles(self):
        self.filePaths = sorted(glob.glob(os.path.join(self.directoryPath, "*.csv")))

    def getModelNames(self):
        for path in self.filePaths:
            filename = path.split('/')[-1]
            match = re.match(r'(.+?)(?:_\d+)?\.csv$', filename)
            if match:
                self.modelNames.append(match.group(1))

    def makeDataFrame(self):
        df_list = []
        for filename in self.filePaths:
               df = pd.read_csv(filename, index_col=None, header=0)
               df_list.append(df)

        self.df = pd.concat(df_list, axis=0, ignore_index=True).rename(columns={'Unnamed: 0':'split'})
        self.df.loc[self.df["split"] == "faint","model_name"] = self.modelNames
        self.df.loc[self.df["split"] == "test","model_name"] = self.modelNames
        self.modelNames = list(set(self.modelNames))

    def getResults(self):
        return self.df[self.df["split"] == "test"].groupby("model_name").mean(numeric_only=True),self.df[self.df["split"] == "faint"].groupby("model_name").mean(numeric_only=True)

        

    