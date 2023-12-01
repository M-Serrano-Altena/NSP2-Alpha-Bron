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
        def exp_gauss(x, amplitude, mu, sigma, labda, gauss1_amplitude, gauss1_mu, gauss1_sigma, gauss2_amplitude, gauss2_mu, gauss2_sigma):
            gauss1 = gauss1_amplitude* sp.stats.norm(loc=gauss1_mu, scale=gauss1_sigma).pdf(x)
            gauss2 = gauss2_amplitude* sp.stats.norm(loc=gauss2_mu, scale=gauss2_sigma).pdf(x)
            exp_gauss = amplitude * sp.stats.exponnorm.pdf(x, K=1/(sigma*labda), loc=mu, scale=sigma)
            return exp_gauss + gauss1 + gauss2
        
        self.model_signal = models.Model(exp_gauss, nan_policy='propagate')
        self.model_signal.set_param_hint('sigma', min=0)
        self.model_signal.set_param_hint('labda', min=0)
        self.result_signal = self.model_signal.fit(self.df_diagram['y0000'], x=self.df_diagram['x0000'], weights=1/self.df_diagram['error_counts'], amplitude=100, mu=start_expmu, sigma = 0.05, labda = 10, gauss1_amplitude=20, gauss1_mu=start_gauss1_mu, gauss1_sigma=0.005, gauss2_amplitude=10, gauss2_mu=start_gauss2_mu, gauss2_sigma=0.005)
        print(self.result_signal.fit_report())
        self.fit_mu = self.result_signal.params['mu'].value
        self.fit_mu_err = self.result_signal.params['mu'].stderr
        self.fit_gauss1mu = self.result_signal.params['gauss1_mu'].value
        self.fit_gauss1mu_err = self.result_signal.params['gauss1_mu'].stderr
        self.fit_gauss2mu = self.result_signal.params['gauss2_mu'].value
        self.fit_gauss2mu_err = self.result_signal.params['gauss2_mu'].stderr

    # fit function
    @staticmethod
    def line(x, a, b):
            return a*x + b
    
    @classmethod
    def energy_volt_fit(cls, volt_setting):
        cls.volt_inv_err = [1/num for num in cls.volt_err]
        cls.model_volt = models.Model(Measurement.line)
        cls.result_volt = cls.model_volt.fit(cls.volt, x=cls.energy_list, a=1, b=0, weights=cls.volt_inv_err)
        print(cls.result_volt.fit_report())
        cls.a = cls.result_volt.params['a'].value
        cls.b = cls.result_volt.params['b'].value

        plt.figure(f'(U,E) diagram {volt_setting} V')
        plt.title(f'(U,E) diagram {volt_setting} V')
        plt.xlabel('Energy (MeV)')
        plt.ylabel('Voltage (V)')
        plt.errorbar(cls.energy_list, cls.volt, yerr= cls.volt_err, fmt='bo', ecolor='k', label='gamma rays')
        plt.plot(cls.energy_list, cls.result_volt.best_fit, 'r-', label='best fit')
        plt.legend(loc='upper left')
        plt.savefig(f'(U,E) diagram {volt_setting} V')
        plt.show()
    
    @classmethod
    def class_energy_volt_diagram(cls, self):
        if "natrium" in self.title.casefold():
            cls.volt.append(self.fit_gauss1mu)
            cls.volt_err.append(self.fit_gauss1mu_err)
            cls.energy_list.append(0.511)
            cls.volt.append(self.fit_gauss2mu)
            cls.volt_err.append(self.fit_gauss2mu_err)
            cls.energy_list.append(1.2)

        elif "cesium" in self.title.casefold():
            cls.volt.append(self.fit_mu)
            cls.volt_err.append(self.fit_mu_err)
            cls.energy_list.append(0.662)

        cls.energy_volt_fit(cls.volt_setting)

    @classmethod
    def volt_to_energy(cls, voltage):
        # y = ax + b --> x = (y-b)/a
        return  (voltage - cls.b)/cls.a
        
        
class Measurement800(Measurement):
    volt = []
    volt_err = []
    energy_list = []
    volt_setting = 800

    def __init__(self, data_file, end_point):
        super().__init__(data_file, end_point)
    
    def energy_volt_diagram(self):
        __class__.class_energy_volt_diagram(self)


class Measurement1000(Measurement):
    volt = []
    volt_err = []
    energy_list = []
    volt_setting = 1000

    def __init__(self, data_file, end_point):
        super().__init__(data_file, end_point)
    
    def energy_volt_diagram(self):
        __class__.class_energy_volt_diagram(self)


meas_vacuum = Measurement("alpha_lucht_100.csv", end_point=1000)
meas_vacuum.data_fit(start_expmu=200, start_gauss1_mu=200, start_gauss2_mu=200)
meas_vacuum.plot()
