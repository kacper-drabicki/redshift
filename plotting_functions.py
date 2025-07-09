import matplotlib.pyplot as plt
import seaborn as sns 
import random

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