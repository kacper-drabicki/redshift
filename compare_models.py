import os
import re
import glob
import pandas as pd

class CompareModels:
    def __init__(self, directoryPath):
        self.directoryPath = directoryPath

        self.filePaths_by_split = {
            "all": [],
            "no_missing": [],
            "only_missing": []
        }

        self.df_by_split = {}
        self.results_by_split = {}

        self.loadFiles()
        self.makeDataFrames()

    def loadFiles(self):
        all_files = sorted(glob.glob(os.path.join(self.directoryPath, "*.csv")))
        for f in all_files:
            if f.endswith("_all.csv"):
                self.filePaths_by_split["all"].append(f)
            elif f.endswith("_no_missing.csv"):
                self.filePaths_by_split["no_missing"].append(f)
            elif f.endswith("_only_missing.csv"):
                self.filePaths_by_split["only_missing"].append(f)

    def makeDataFrames(self):
        def load_group(file_list):
            dfs = []
            for path in file_list:
                filename = os.path.basename(path)
                # Capture everything before the _all / _no_missing / _only_missing suffix
                model_name = re.match(r'(.+?)_(?:all|no_missing|only_missing)\.csv$', filename).group(1)
                df = pd.read_csv(path, index_col=None, header=0)
                df = df.rename(columns={'Unnamed: 0': 'split'}, errors="ignore")
                df["model_name"] = model_name
                dfs.append(df)
            if dfs:
                return pd.concat(dfs, axis=0, ignore_index=True)
            else:
                return pd.DataFrame()

        for split_type, file_list in self.filePaths_by_split.items():
            self.df_by_split[split_type] = load_group(file_list)

    def getResults(self):
        def group_results(df):
            if df.empty:
                return pd.DataFrame(), pd.DataFrame()
            return (
                df[df["split"] == "test"].groupby("model_name").mean(numeric_only=True),
                df[df["split"] == "faint"].groupby("model_name").mean(numeric_only=True)
            )

        for split_type, df in self.df_by_split.items():
            self.results_by_split[split_type] = group_results(df)

        return self.results_by_split

    