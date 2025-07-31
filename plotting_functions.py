import matplotlib.pyplot as plt
import random
import numpy as np

def diag_plot(test_Z, test_Z_pred, test_Z_pred_std, faint_Z, faint_Z_pred, faint_Z_pred_std):
    random.seed(1)
        
    combined = list(zip(test_Z, test_Z_pred, test_Z_pred_std))
    sampled = random.sample(combined, 500)
    y_test, y_pred, y_pred_std = zip(*sampled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
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
    ax1.grid(True)
    
    cbar1 = fig.colorbar(sc1, ax=ax1)
    cbar1.set_label("Z$_{photo}$ std. dev.")
  
    combined = list(zip(faint_Z, faint_Z_pred, faint_Z_pred_std))
    sampled = random.sample(combined, 1000)
    faint_test_sampled, faint_pred_sampled, faint_pred_std_sampled = zip(*sampled)
    
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
    ax2.grid(True)
    
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label("Z$_{photo}$ std. dev.")
    
    plt.tight_layout()

    return fig

def dist_plot(x_input, true_y, model, points=np.arange(0, 6, 0.01)):
    r = x_input[2]
    y_model = model(x_input)
    y_pred = y_model.mean().numpy()

    probs = []
    for point in points:
        log_prob = y_model.log_prob([[point]])
        probs.append(np.exp(log_prob))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(points, probs, color='steelblue')
    ax.axvline(true_y, color='red', linestyle='-', label='True value')
    ax.axvline(y_pred, color='black', linestyle='--', label='Pred value')
    ax.set_xlim(-0.05, 4.5)
    ax.legend()
    
    plt.tight_layout()
    
    return fig
    