import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os
import re
from collections import defaultdict

class CalculateMetrics:
    def __init__(self, data_dir: str, output_dir: str, verbose: bool = False):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.verbose = verbose
        self.mse = mean_squared_error
        self.r2 = r2_score

    def _log(self, *args):
        if self.verbose:
            print(*args)

    def _group_files_by_model(self):
        """Group all CSV files in the folder by model name (before last _<int>)."""
        files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        groups = defaultdict(list)

        for f in files:
            match = re.match(r"(.+)_\d+\.csv", f)
            if match:
                model_name = match.group(1)
                groups[model_name].append(f)

        return groups

    def _load_group_data(self, file_list):
        dfs = []
        for filename in sorted(file_list):  # keep consistent order
            path = os.path.join(self.data_dir, filename)
            df = pd.read_csv(path, index_col=0)

            cols = ["Z", "Z_pred", "faint", "has_missing"]
            if "Z_pred_std" in df.columns:
                cols.append("Z_pred_std")
            if "Z_spec_prob" in df.columns:
                cols.append("Z_spec_prob")

            df = df.loc[df["split"] == "test", cols]
            dfs.append(df)
        return dfs

    # Ensemble
    # def _compute_metrics(self, dfs, subset: str = "all"):
    #     test_mses, test_r2s = [], []
    #     faint_mses, faint_r2s = [], []
    #     test_probs, faint_probs = [], []

    #     for df in dfs:
    #         if subset == "all":
    #             test_df = df.loc[~df["faint"]]
    #             faint_df = df.loc[df["faint"]]
    #         elif subset == "no_missing":
    #             test_df = df.loc[(~df["faint"]) & (~df["has_missing"])]
    #             faint_df = df.loc[(df["faint"]) & (~df["has_missing"])]
    #         elif subset == "only_missing":
    #             test_df = df.loc[(~df["faint"]) & (df["has_missing"])]
    #             faint_df = df.loc[(df["faint"]) & (df["has_missing"])]
    #         else:
    #             raise ValueError("subset must be 'all', 'no_missing', or 'only_missing'")

    #         if len(test_df) > 0:
    #             test_mses.append(self.mse(test_df["Z"], test_df["Z_pred"]))
    #             test_r2s.append(self.r2(test_df["Z"], test_df["Z_pred"]))
    #             if "Z_spec_prob" in df.columns:
    #                 test_probs.append(test_df["Z_spec_prob"])
    #         else:
    #             test_mses.append(np.nan)
    #             test_r2s.append(np.nan)

    #         if len(faint_df) > 0:
    #             faint_mses.append(self.mse(faint_df["Z"], faint_df["Z_pred"]))
    #             faint_r2s.append(self.r2(faint_df["Z"], faint_df["Z_pred"]))
    #             if "Z_spec_prob" in df.columns:
    #                 faint_probs.append(faint_df["Z_spec_prob"])
    #         else:
    #             faint_mses.append(np.nan)
    #             faint_r2s.append(np.nan)

    #     results = {
    #         "test": {
    #             "MSE": np.nanmean(test_mses),
    #             "R^2": np.nanmean(test_r2s),
    #             "NLL": None,
    #         },
    #         "faint": {
    #             "MSE": np.nanmean(faint_mses),
    #             "R^2": np.nanmean(faint_r2s),
    #             "NLL": None,
    #         },
    #     }

    #     if test_probs and faint_probs:
    #         epsilon = 1E-38
    #         test_prob_array = np.array([prob.values for prob in test_probs])
    #         faint_prob_array = np.array([prob.values for prob in faint_probs])

    #         test_mean_prob = np.clip(test_prob_array.mean(axis=0), epsilon, None)
    #         faint_mean_prob = np.clip(faint_prob_array.mean(axis=0), epsilon, None)

    #         results["test"]["NLL"] = -np.log(test_mean_prob).mean()
    #         results["faint"]["NLL"] = -np.log(faint_mean_prob).mean()

    #     return results

    def _compute_metrics(self, dfs, subset: str = "all"):
        test_mses, test_r2s = [], []
        faint_mses, faint_r2s = [], []
        test_probs, faint_probs = [], []
    
        for df in dfs:
            if subset == "all":
                test_df = df.loc[~df["faint"]]
                faint_df = df.loc[df["faint"]]
            elif subset == "no_missing":
                test_df = df.loc[(~df["faint"]) & (~df["has_missing"])]
                faint_df = df.loc[(df["faint"]) & (~df["has_missing"])]
            elif subset == "only_missing":
                test_df = df.loc[(~df["faint"]) & (df["has_missing"])]
                faint_df = df.loc[(df["faint"]) & (df["has_missing"])]
            else:
                raise ValueError("subset must be 'all', 'no_missing', or 'only_missing'")
    
            if len(test_df) > 0:
                test_mses.append(self.mse(test_df["Z"], test_df["Z_pred"]))
                test_r2s.append(self.r2(test_df["Z"], test_df["Z_pred"]))
                if "Z_spec_prob" in df.columns:
                    test_probs.append(test_df["Z_spec_prob"])
            else:
                test_mses.append(np.nan)
                test_r2s.append(np.nan)
    
            if len(faint_df) > 0:
                faint_mses.append(self.mse(faint_df["Z"], faint_df["Z_pred"]))
                faint_r2s.append(self.r2(faint_df["Z"], faint_df["Z_pred"]))
                if "Z_spec_prob" in df.columns:
                    faint_probs.append(faint_df["Z_spec_prob"])
            else:
                faint_mses.append(np.nan)
                faint_r2s.append(np.nan)
    
        results = {
            "test": {
                "MSE": np.nanmean(test_mses),
                "R^2": np.nanmean(test_r2s),
                "NLL_mean": None,
            },
            "faint": {
                "MSE": np.nanmean(faint_mses),
                "R^2": np.nanmean(faint_r2s),
                "NLL_mean": None,
            },
        }
    
        if test_probs and faint_probs:
            epsilon = 1E-38
            test_prob_array = np.array([prob.values for prob in test_probs])
            faint_prob_array = np.array([prob.values for prob in faint_probs])
    
            # Per-run NLLs
            test_run_nlls = [-np.log(np.clip(arr, epsilon, None)).mean()
                             for arr in test_prob_array]
            faint_run_nlls = [-np.log(np.clip(arr, epsilon, None)).mean()
                              for arr in faint_prob_array]
    
            # Average across runs
            results["test"]["NLL_mean"] = np.mean(test_run_nlls)
            results["faint"]["NLL_mean"] = np.mean(faint_run_nlls)
    
        return results

    def _save_results(self, results, model_name, suffix):
        df_results = pd.DataFrame(results).T
        filename = os.path.join(self.output_dir, f"{model_name}{suffix}.csv")
        df_results.to_csv(filename)
        self._log(f"Saved results to {filename}")

    def run_all(self):
        """Run evaluation for every model group found in data_dir."""
        groups = self._group_files_by_model()
        self._log(f"Found {len(groups)} model groups.")

        for model_name, file_list in groups.items():
            self._log(f"\nProcessing model: {model_name}")
            dfs = self._load_group_data(file_list)

            for subset, suffix in [
                ("all", "_all"),
                ("no_missing", "_no_missing"),
                ("only_missing", "_only_missing"),
            ]:
                results = self._compute_metrics(dfs, subset=subset)
                self._save_results(results, model_name, suffix)