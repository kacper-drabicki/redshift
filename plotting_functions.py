import matplotlib.pyplot as plt
import seaborn as sns 
import random
import pandas as pd
import numpy as np
from metrics import redshift_error
from sklearn.metrics import mean_squared_error, r2_score

def plotTrainHistory(history, val=True):
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].plot(history.history['loss'], label="train")
    if val:
        axes[0].plot(history.history['val_loss'], label="val")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label="train")
    if val:
        axes[1].plot(history.history['val_loss'], label="val")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_yscale('log')
    axes[1].legend()

    plt.subplots_adjust(wspace=0.5)

def redshift_plot(y_true, y_pred):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Scatter Plot
    paired = list(zip(y_true, y_pred))
    sampled_pairs = random.sample(paired, 1000)
    y_true_sample, y_pred_sample = zip(*sampled_pairs)
   
    sns.scatterplot(x=y_true_sample, y=y_pred_sample, s=10, color="red", alpha=0.2, ax=axes[0])
    sns.lineplot(x=y_true, y=y_true, color="blue", ax=axes[0])
    axes[0].set_xlabel("Z$_{spec}$", fontsize=12)
    axes[0].set_ylabel("Z$_{photo}$", fontsize=12)

    # KDE Plot
    sns.kdeplot(x=y_true, y=y_pred, fill=True, ax=axes[1])
    sns.lineplot(x=y_true, y=y_true, color="red", ax=axes[1])
    axes[1].set_xlabel("Z$_{spec}$", fontsize=12)
    axes[1].set_ylabel("Z$_{photo}$", fontsize=12)

    plt.show()

def plot_rmag_vs_metric(df, model):
    df['MAG_bin'] = pd.cut(df['MAG_GAAP_r'], bins=np.arange(15.38, 25.96, 0.1), include_lowest=True)
    
    mse_per_bin = {}
    r2_per_bin = {}
    redshift_err_per_bin = {}
    
    
    for bin_label, group in df.groupby('MAG_bin'):
        if len(group) < 2:
            continue
    
        X_bin = group.drop(columns=['Z', 'MAG_bin'])
        y_bin = group['Z']
    
        y_pred = model.predict(X_bin)
        r2 = r2_score(y_bin, y_pred)
        r2_per_bin[str(bin_label)] = r2
        mse = mean_squared_error(y_bin, y_pred)
        mse_per_bin[str(bin_label)] = mse
        redshift_err = redshift_error(y_bin, y_pred)
        redshift_err_per_bin[str(bin_label)] = redshift_err[0]

    metrics_df = pd.DataFrame({
    'MAG_bin': list(r2_per_bin.keys()),
    'R2': list(r2_per_bin.values()),
    'MSE': list(mse_per_bin.values()),
    'Redshift_Error': list(redshift_err_per_bin.values())
})

    def bin_center(interval_str):
        left, right = interval_str.strip('()[]').split(',')
        return (float(left) + float(right)) / 2
    
    metrics_df['bin_center'] = metrics_df['MAG_bin'].apply(bin_center)
    metrics_df = metrics_df.sort_values('bin_center')
    
    fig, axs = plt.subplots(3, 1, figsize=(7, 10))
    
    # Plot R²
    axs[0].scatter(metrics_df['bin_center'], metrics_df['R2'], color='blue')
    axs[0].set_ylim(0,1)
    axs[0].set_xlabel('MAG_GAAP_r')
    axs[0].set_ylabel('R² Score')
    axs[0].grid(True)
    
    # Plot MSE
    axs[1].scatter(metrics_df['bin_center'], metrics_df['MSE'], color='red')
    axs[1].set_xlabel('MAG_GAAP_r')
    axs[1].set_ylabel('Mean Squared Error')
    axs[1].grid(True)
    
    # Plot Redshift Error
    axs[2].scatter(metrics_df['bin_center'], metrics_df['Redshift_Error'], color='green')
    axs[2].set_xlabel('MAG_GAAP_r')
    axs[2].set_ylabel('Redshift Error')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()