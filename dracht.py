import pandas as pd
import matplotlib.pyplot as plt
from lmfit import models
import numpy as np
import scipy as sp


class Measurement:
    volt = []
    energy_list = []

    def __init__(self, data_file, end_point):
        self.data_file = data_file
        self.df_diagram = pd.read_csv(data_file)
        self.df_diagram = self.df_diagram.loc[(self.df_diagram['x0000'] < end_point) & (self.df_diagram['y0000'] > 0)]
        self.df_diagram['x0000'] /= 1000 # from mV to V
        self.df_diagram["error_counts"] = np.sqrt(self.df_diagram['y0000'])
        self.title, _ = self.data_file.split('.')
        
        
    def plot(self):
        fig = plt.figure(self.title)
        plt.title(self.title.replace("_", ' '))
        plt.errorbar(self.df_diagram['x0000'], self.df_diagram['y0000'], yerr=self.df_diagram['error_counts'], fmt='bo', ecolor='k', label='data')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Counts')
        plt.plot(self.df_diagram['x0000'], self.result_signal.best_fit, 'r', label='best fit')
        plt.legend(loc='upper right')
        plt.savefig(f"{self.title}.png")
        plt.show()
    
    def data_fit(self, start_expmu, start_gauss1_mu, start_gauss2_mu):
        def exp_gauss(x, amplitude, mu, sigma, labda, gauss1_amplitude, gauss1_mu, gauss1_sigma):
            gauss1 = gauss1_amplitude* sp.stats.norm(loc=gauss1_mu, scale=gauss1_sigma).pdf(x)
            gauss2 = gauss2_amplitude* sp.stats.norm(loc=gauss2_mu, scale=gauss2_sigma).pdf(x)
            exp_gauss = amplitude * sp.stats.exponnorm.pdf(x, K=1/(sigma*labda), loc=mu, scale=sigma)
            return exp_gauss + gauss1 + gauss2
        
        self.model_signal = models.Model(exp_gauss, nan_policy='propagate')
        self.model_signal.set_param_hint('sigma', min=0)
        self.model_signal.set_param_hint('labda', min=0)
        self.result_signal = self.model_signal.fit(self.df_diagram['y0000'], x=self.df_diagram['x0000'], weights=1/self.df_diagram['error_counts'], amplitude=100, mu=start_expmu, sigma = 0.05, labda = 10, gauss1_amplitude=20, gauss1_mu=start_gauss1_mu, gauss1_sigma=0.005)
        print(self.result_signal.fit_report())
        self.fit_mu = self.result_signal.params['mu'].value
        self.fit_mu_err = self.result_signal.params['mu'].stderr
        self.fit_gauss1mu = self.result_signal.params['gauss1_mu'].value
        self.fit_gauss1mu_err = self.result_signal.params['gauss1_mu'].stderr
        self.fit_gauss2mu = self.result_signal.params['gauss2_mu'].value
        self.fit_gauss2mu_err = self.result_signal.params['gauss2_mu'].stderr

    @staticmethod
    def volt_to_energy(voltage, a):
        # y = ax + b --> x = (y-b)/a
        return  (voltage)/a
        


meas_vacuum = Measurement("alfa bron 21 mbar.csv", end_point=1000)
meas_vacuum.data_fit(start_expmu=200, start_gauss1_mu=200, start_gauss2_mu=200)
meas_vacuum.plot()
