import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from metrics import redshift_error
from sklearn.metrics import mean_squared_error, r2_score

class Evaluator():
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.test_data = self.dataFrame.data[self.dataFrame.data["split"] == "test"]
        # self.test_data = self.test_data.loc[self.test_data["MAG_GAAP_r"] < 25.0]
        self.y_test = self.test_data.loc[~self.test_data["faint"], "Z"]
        self.y_pred = self.test_data.loc[~self.test_data["faint"], "Z_pred"]
        self.y_pred_std = self.test_data.loc[~self.test_data["faint"], "Z_pred_std"]
        self.test_log_prob = self.test_data.loc[~self.test_data["faint"], "log_prob"]
        self.faint_test = self.test_data.loc[self.test_data["faint"], "Z"]
        self.faint_pred = self.test_data.loc[self.test_data["faint"], "Z_pred"]
        self.faint_pred_std = self.test_data.loc[self.test_data["faint"], "Z_pred_std"]
        self.faint_log_prob = self.test_data.loc[self.test_data["faint"], "log_prob"]
        
        self.mse = mean_squared_error
        self.r2_score = r2_score
        self.redshift_error = redshift_error

    def evaluate_metrics(self):
        return pd.DataFrame({"test":
                {"MSE": self.mse(self.y_test, self.y_pred),
                "R^2": self.r2_score(self.y_test, self.y_pred),
                "Redshift error": self.redshift_error(self.y_test, self.y_pred),
                "NLL": self.test_log_prob.mean()},
                "faint":
                {"MSE": self.mse(self.faint_test, self.faint_pred),
                "R^2": self.r2_score(self.faint_test, self.faint_pred),
                "Redshift error": self.redshift_error(self.faint_test, self.faint_pred),
                "NLL": self.faint_log_prob.mean()}
               }).T

    def redshift_std(self):
        random.seed(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        combined = list(zip(self.y_test, self.y_pred, self.y_pred_std))
        sampled = random.sample(combined, 500)
        y_test, y_pred, y_pred_std = zip(*sampled)


        sc1 = ax1.scatter(y_test, y_pred,
                          c=y_pred_std,
                          cmap='rainbow',
                          s=20,
                          alpha=0.5,
                          vmin=0,
                          vmax=1.2,
                         )
        ax1.plot([0,4],
                 [0,4],
                 'r')
        ax1.set_xlim(0,4)
        ax1.set_ylim(0,4)
        ax1.set_xlabel("Z$_{spec}$")
        ax1.set_ylabel("Z$_{photo}$")
        ax1.set_title(f"{self.dataFrame.objtype}$_{{spec}}$")
        ax1.grid(True)
        
        cbar1 = fig.colorbar(sc1, ax=ax1)
        cbar1.set_label("Z$_{photo}$ std. dev.")

        combined = list(zip(self.faint_test, self.faint_pred, self.faint_pred_std))
        sampled = random.sample(combined, 1000,)
        faint_test, faint_pred, faint_pred_std = zip(*sampled)

        sc2 = ax2.scatter(faint_test, faint_pred,
                          c=faint_pred_std,
                          cmap='rainbow',
                          s=20,
                          alpha=0.5,
                          vmin=0,
                          vmax=1.2,
                         )
        ax2.plot([0,4],
                 [0,4],
                 'r')
        ax2.set_xlim(0,4)
        ax2.set_ylim(0,4)
        ax2.set_xlabel("Z$_{spec}$")
        ax2.set_ylabel("Z$_{photo}$")
        ax2.set_title(f"{self.dataFrame.objtype}$_{{spec}}$")
        ax2.grid(True)
        cbar2 = fig.colorbar(sc2, ax=ax2)
        cbar2.set_label("Z$_{photo}$ std. dev.")

        plt.tight_layout()

        plt.savefig("../../plots/plot.png")
        plt.show()

        

    def redshift_kde(self):
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        
        # Scatter Plot
        paired = list(zip(self.y_test, self.y_pred))
        sampled_pairs = random.sample(paired, 1000)
        y_true_sample, y_pred_sample = zip(*sampled_pairs)
        
        sns.scatterplot(x=y_true_sample, y=y_pred_sample, s=10, color="red", alpha=0.2, ax=axes[0])
        sns.lineplot(x=self.y_test, y=self.y_test, color="blue", ax=axes[0])
        axes[0].set_xlabel("Z$_{spec}$", fontsize=12)
        axes[0].set_ylabel("Z$_{photo}$", fontsize=12)
        
        # KDE Plot
        sns.kdeplot(x=self.y_test, y=self.y_pred, fill=True, ax=axes[1])
        sns.lineplot(x=self.y_test, y=self.y_test, color="red", ax=axes[1])
        axes[1].set_xlabel("Z$_{spec}$", fontsize=12)
        axes[1].set_ylabel("Z$_{photo}$", fontsize=12)
        
        plt.show()

    