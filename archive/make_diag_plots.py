import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

models = ["single","double","triple"]

for model in [1]:

    dataframes = []
    for i in range(1,6):
        df = pd.read_csv(f"../dataframes/experiment2.1/MG_3_components_{i}.csv", index_col=0)
        df = df.loc[df["split"]=="test",["Z","Z_pred","Z_pred_std","faint"]]
        dataframes.append(df)

    test_Z = dataframes[0].loc[~dataframes[0]["faint"], "Z"]
    test_Z_pred = []
    test_Z_pred_std = []
    faint_Z = dataframes[0].loc[dataframes[0]["faint"], "Z"]
    faint_Z_pred = []
    faint_Z_pred_std = []

    for df in dataframes:
        test_df = df.loc[~df["faint"]]
        faint_df = df.loc[df["faint"]]

        test_Z_pred.append(test_df["Z_pred"])
        test_Z_pred_std.append(test_df["Z_pred_std"])
        faint_Z_pred.append(faint_df["Z_pred"])
        faint_Z_pred_std.append(faint_df["Z_pred_std"])

    test_Z_pred = np.array(test_Z_pred).mean(axis=0)
    test_Z_pred_std = np.sqrt(np.sum(np.array(test_Z_pred_std)**2 / 5, axis=0))

    faint_Z_pred = np.array(faint_Z_pred).mean(axis=0)
    faint_Z_pred_std = np.sqrt(np.sum(np.array(faint_Z_pred_std)**2 / 5, axis=0))

    random.seed(1)
        
    combined = list(zip(test_Z, test_Z_pred, test_Z_pred_std))
    sampled = random.sample(combined, 500)
    y_test, y_pred, y_pred_std = zip(*sampled)
    
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sc1 = ax1.scatter(y_test, y_pred,
                      c=y_pred_std,
                      cmap='inferno',
                      s=20,
                      alpha=0.5,
                      vmin=0,
                      vmax=1.2,
                     )
    ax1.plot([0,4], [0,4], 'r')
    ax1.set_xlim(0,4)
    ax1.set_ylim(0,4)
    ax1.set_xlabel("Z$_{spec}$")
    ax1.set_ylabel("Z$_{photo}$")
    ax1.set_title(f"gaussian on random test dataset")
    ax1.grid(True)
    
    cbar1 = fig1.colorbar(sc1, ax=ax1)
    cbar1.set_label("Z$_{photo}$ std. dev.")
    
    plt.tight_layout()
    fig1.savefig(f'../plots/{model}_test.png')
    plt.close(fig1)
  
    combined = list(zip(faint_Z, faint_Z_pred, faint_Z_pred_std))
    sampled = random.sample(combined, 1000)
    faint_test_sampled, faint_pred_sampled, faint_pred_std_sampled = zip(*sampled)
    
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sc2 = ax2.scatter(faint_test_sampled, faint_pred_sampled,
                      c=faint_pred_std_sampled,
                      cmap='inferno',
                      s=20,
                      alpha=0.5,
                      vmin=0,
                      vmax=1.2,
                     )
    ax2.plot([0,4], [0,4], 'r')
    ax2.set_xlim(0,4)
    ax2.set_ylim(0,4)
    ax2.set_xlabel("Z$_{spec}$")
    ax2.set_ylabel("Z$_{photo}$")
    ax2.set_title(f"gaussian on faint test dataset")
    ax2.grid(True)
    
    cbar2 = fig2.colorbar(sc2, ax=ax2)
    cbar2.set_label("Z$_{photo}$ std. dev.")
    
    plt.tight_layout()
    fig2.savefig(f'../plots/{model}_faint.png')
    plt.close(fig2)