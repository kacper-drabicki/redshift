import matplotlib.pyplot as plt
import random

def make_diag_plot(Z, Z_pred, Z_pred_std):
    random.seed(1)
        
    combined = list(zip(Z, Z_pred, Z_pred_std))
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
    ax1.set_title(f"{model} gaussian on random test dataset")
    ax1.grid(True)
    
    cbar1 = fig1.colorbar(sc1, ax=ax1)
    cbar1.set_label("Z$_{photo}$ std. dev.")
    
    plt.tight_layout()
    fig1.savefig(f'../plots/{model}_test.png')
    plt.close(fig1)