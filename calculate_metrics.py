import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

models = ["single","double","triple"]

for model in models:
    
    dataframes = []
    for i in range(1,6):
        df = pd.read_csv(f"../dataframes/{model}_gauss_{i}", index_col=0)
        df = df.loc[df["split"]=="test",["Z","Z_pred","Z_pred_std","Z_spec_prob","faint"]]
        dataframes.append(df)
    
    mse = mean_squared_error
    r2 = r2_score
    
    test_mean_squared_errors = []
    test_r2_scores = []
    faint_mean_squared_errors = []
    faint_r2_scores = []
    test_probs = []
    faint_probs = []
    
    for df in dataframes:
        test_df = df.loc[~df["faint"]]
        faint_df = df.loc[df["faint"]]
        
        test_mean_squared_errors.append(mse(test_df["Z"], test_df["Z_pred"]))
        test_r2_scores.append(r2(test_df["Z"], test_df["Z_pred"]))
        faint_mean_squared_errors.append(mse(faint_df["Z"], faint_df["Z_pred"]))
        faint_r2_scores.append(r2(faint_df["Z"], faint_df["Z_pred"]))
    
        test_probs.append(test_df["Z_spec_prob"])
        faint_probs.append(faint_df["Z_spec_prob"])
    
    test_mean_squared_error = np.array(test_mean_squared_errors).mean()
    test_r2_score = np.array(test_r2_scores).mean()
    faint_mean_squared_error = np.array(faint_mean_squared_errors).mean()
    faint_r2_score = np.array(faint_r2_scores).mean()
    
    epsilon = 1E-38
    test_NLL, faint_NLL = -np.log(np.clip(np.array(test_probs).mean(axis=0), epsilon, None)).mean(), -np.log(np.clip(np.array(faint_probs).mean(axis=0), epsilon, None)).mean()
    # test_NLL, faint_NLL = -np.log(np.array(test_probs).mean(axis=0)).mean(), -np.log(np.array(faint_probs).mean(axis=0)).mean()
    
    results = pd.DataFrame({"test":
                    {"MSE": test_mean_squared_error,
                    "R^2": test_r2_score,
                    "NLL": test_NLL},
                    "faint":
                    {"MSE": faint_mean_squared_error,
                    "R^2": faint_r2_score,
                    "NLL": faint_NLL}
                   }).T
    
    results.to_csv(f"../metrics/{model}_gauss.csv")