import numpy as np
import matplotlib.pyplot as plt

BINS = np.arange(0, 5, 0.04)
BIN_CENTERS = 0.5 * (BINS[1:] + BINS[:-1])
SAMPLES_N = 1
HISTOGRAM_N = 5000
BURN_IN = 10

class HistPlotter:

    def __init__(self, X, y, datasetType, model, modelName, BNN=False):
        self.X = X
        self.y = y
        self.datasetType = datasetType
        self.model = model
        self.modelName = modelName
        self.BNN = BNN

    def calculate_hist(self):
        
        histograms = []
        self.global_std_history = []
        
        if not self.BNN: # model is deterministic
            distributions = self.model(self.X)
    
        for i in range(HISTOGRAM_N):
            if self.BNN: # model is stochastic
                distributions = self.model(self.X)
            samples = distributions.sample(SAMPLES_N).numpy()
            hist, _ = np.histogram(samples.reshape(-1), bins=BINS, density=True)
            histograms.append(hist)
    
            if i >= BURN_IN:
                arr = np.array(histograms)
                var_per_bin = arr.var(axis=0, ddof=1)
                global_std = np.sqrt(var_per_bin.mean())
                self.global_std_history.append(global_std)
    
        histograms = np.array(histograms)
        mean = np.mean(histograms, axis=0)
        std  = np.std(histograms, axis=0)
    
        return mean, std
        
    def calculate_chi_square(self, mean, std, y):

        error = self.error_model(mean)
        
        hist, _ = np.histogram(y, bins=BINS, density=True)
        
        diff = (hist - mean).reshape(-1,1)
        cov = np.diag(error ** 2)
        cov_inv = np.linalg.inv(cov)
        chi_square = diff.T @ cov_inv @ diff
    
        return chi_square.item()

    def plot_pdf(self, mean, std, title="Distribution"):
        
        plt.figure(figsize=(9,7.5))
        plt.plot(BIN_CENTERS, mean, drawstyle='steps-mid',  color="orange", label=f'Pred')
        plt.plot(BIN_CENTERS, self.error_model(mean), label="Model error")
        plt.fill_between(BIN_CENTERS, mean-std, mean+std, color="orange", alpha=0.3)
        plt.hist(self.y, bins=BINS, histtype="step", density=True, color="blue", linestyle="--",label="True")
        plt.xlabel('Redshift Z')
        plt.ylabel('PDF')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_std_history(self):
        
        plt.figure(figsize=(9,7.5))
        plt.plot(np.arange(BURN_IN, HISTOGRAM_N,1), self.global_std_history, label=self.modelName)
        plt.xlabel("Number of Histograms")
        plt.ylabel("Mean Error Across Bins")
        plt.legend()
        plt.show()

    def error_model(self, mean):
        return 0.05 * mean + 0.005 * BIN_CENTERS

    def evaluate_model(self):
    
        mean, std = self.calculate_hist()
    
        chi_square = self.calculate_chi_square(mean, std, self.y)
        print("Chi^2:",chi_square)
        
        self.plot_pdf(mean, std, f"{self.modelName} with 3 components - {self.datasetType} test dataset")

    