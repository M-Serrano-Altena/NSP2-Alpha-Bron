"""Fits the measured data with a exponential gauss with an added normal gauss and plots the fit. 
"""

import pandas as pd
import matplotlib.pyplot as plt
from lmfit import models
import numpy as np
import scipy as sp
import os

python_file_path = os.getcwd()

os.chdir(os.path.join(python_file_path, "CSV data"))


class Measurement:
    """Can calculate a fit of a measurement and plot it. Can also convert a voltage to an energy value
    """
    
    volt = []
    energy_list = []

    def __init__(self, data_file: str, end_point: int = None):
        """read the data of given csv file and put it in a dataframe

        Args:
            data_file: name of the csv file to be read
            end_point: end point of the data. Everything after this value in mV will be deleted from the dataframe
        """        
        self.data_file = data_file
        self.df_diagram = pd.read_csv(data_file)
        self.df_diagram = self.df_diagram.loc[(self.df_diagram['y0000'] > 0)]

        if end_point != None:
            self.df_diagram = self.df_diagram.loc[(self.df_diagram['x0000'] < end_point)]
        
        self.df_diagram['x0000'] /= 1000 # from mV to V
        self.df_diagram["error_counts"] = np.sqrt(self.df_diagram['y0000'])
        self.title, _ = self.data_file.split('.')
        
        
    def plot_fit(self):
        """Plots the fit calculated in the data_fit function. Requires the data_fit function to have been run before this function
        """        
        fig = plt.figure(self.title)
        plt.title(self.title.replace("_", ' '))
        plt.errorbar(self.df_diagram['x0000'], self.df_diagram['y0000'], yerr=self.df_diagram['error_counts'], fmt='bo', ecolor='k', label='data')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Counts')
        plt.plot(self.df_diagram['x0000'], self.result_signal.best_fit, 'r', label='best fit')
        plt.legend(loc='upper right')

        # go to fits folder to save fit
        os.chdir(os.path.join(python_file_path, "Fits"))
        plt.savefig(f"{self.title}_fit.png")

        # go to csv folder to read csv
        os.chdir(os.path.join(python_file_path, "CSV data"))
        plt.show()
    
    def data_fit(self, start_expmu: float, start_gauss1_mu: float):
        """fits the measured data with an exponential gauss with an added normal gauss

        Args:
            start_expmu: start mean value of the exponential gauss for the fit
            start_gauss1_mu: start mean value of the normal gauss for the fit
        """        
        def exp_gauss(x: list, amplitude: float, mu: float, sigma: float, labda: float, gauss1_amplitude: float, gauss1_mu: float, gauss1_sigma: float) -> np.ndarray:
            """exponential gauss with an added normal gauss

            Args:
                x: list of x coordinates
                amplitude: the amplitude of the exponential gauss
                mu: the mean value of the exponential gauss
                sigma: the standard devitaion of the exponential gauss. Always greater than zero
                labda: the rate of the exponential component. Always greater than zero
                gauss1_amplitude: the amplitude of the normal gauss
                gauss1_mu: the mean value of the normal gauss
                gauss1_sigma: the standard deviation of the normal gauss

            Returns:
                exponential gauss with an added normal gauss
            """            
            gauss1 = gauss1_amplitude* sp.stats.norm(loc=gauss1_mu, scale=gauss1_sigma).pdf(x)
            exp_gauss = amplitude * sp.stats.exponnorm.pdf(x, K=1/(sigma*labda), loc=mu, scale=sigma)
            return exp_gauss + gauss1
        
        self.model_signal = models.Model(exp_gauss, nan_policy='propagate')
        self.model_signal.set_param_hint('sigma', min=0)
        self.model_signal.set_param_hint('labda', min=0)
        self.result_signal = self.model_signal.fit(self.df_diagram['y0000'], x=self.df_diagram['x0000'], weights=1/self.df_diagram['error_counts'], amplitude=100, mu=start_expmu, sigma = 0.05, labda = 10, gauss1_amplitude=20, gauss1_mu=start_gauss1_mu, gauss1_sigma=0.005)
        print(self.result_signal.fit_report())
        self.fit_mu = self.result_signal.params['mu'].value
        self.fit_mu_err = self.result_signal.params['mu'].stderr
        self.fit_gauss1mu = self.result_signal.params['gauss1_mu'].value
        self.fit_gauss1mu_err = self.result_signal.params['gauss1_mu'].stderr

    @staticmethod
    def volt_to_energy(voltage: float, a: float) -> float:
        """converts a voltage to an energy value

        Args:
            voltage: voltage to be converted
            a: converstion factor

        Returns:
            energy value corresponding to the input voltage
        """        
        # y = ax + b --> x = (y-b)/a
        return  (voltage)/a
        

def run():
    meas1_vacuum = Measurement("alfa bron 21 mbar.csv", end_point=1000)
    meas1_vacuum.data_fit(start_expmu=0.05, start_gauss1_mu=0.2)
    meas1_vacuum.plot_fit()

    meas1_air_100mbar = Measurement("alfa bron lucht 100 mbar.csv", end_point=1000)
    meas1_air_100mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.18)
    meas1_air_100mbar.plot_fit()

    meas1_air_200mbar = Measurement("alfa bron lucht 200 mbar.csv", end_point=1000)
    meas1_air_200mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.2)
    meas1_air_200mbar.plot_fit()

    meas1_air_300mbar = Measurement("alfa bron lucht 300 mbar.csv", end_point=1000)
    meas1_air_300mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.18)
    meas1_air_300mbar.plot_fit()

    meas1_air_400mbar = Measurement("alfa bron lucht 400 mbar.csv", end_point=1000)
    meas1_air_400mbar.data_fit(start_expmu=0.025, start_gauss1_mu=0.17)
    meas1_air_400mbar.plot_fit()

    meas1_air_500mbar = Measurement("alfa bron lucht 500 mbar.csv", end_point=1000)
    meas1_air_500mbar.data_fit(start_expmu=0.025, start_gauss1_mu=0.16)
    meas1_air_500mbar.plot_fit()

    meas1_air_600mbar = Measurement("alfa bron lucht 600 mbar.csv", end_point=1000)
    meas1_air_600mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.14)
    meas1_air_600mbar.plot_fit()

    meas1_air_700mbar = Measurement("alfa bron lucht 700 mbar.csv", end_point=1000)
    meas1_air_700mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    meas1_air_700mbar.plot_fit()

    meas1_air_800mbar = Measurement("alfa bron lucht 800 mbar.csv", end_point=1000)
    meas1_air_800mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.12)
    meas1_air_800mbar.plot_fit()

    meas1_air_900mbar = Measurement("alfa bron lucht 900 mbar.csv", end_point=1000)
    meas1_air_900mbar.data_fit(start_expmu=0.022, start_gauss1_mu=0.12)
    meas1_air_900mbar.plot_fit()


if __name__ == "__main__":
    run()