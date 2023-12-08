"""Fits the measured data with a exponential gauss with an added normal gauss and plots the fit. 
"""

import pandas as pd
import matplotlib.pyplot as plt
from lmfit import models
import numpy as np
import scipy as sp
import os

path = os.path.dirname(__file__)
path = os.path.dirname(path)
os.chdir(os.path.dirname(path))

python_file_path = os.getcwd()

os.chdir(os.path.join(python_file_path, "CSV data"))


class Measurement:
    """Can calculate a fit of a measurement and plot it. Can also convert a voltage to an energy value
    """

    pressure_list = []
    path_length_list = []

    volt = []
    volt_error = []

    conversion_factor = 1 # placeholder value
    energy_alpha_init = 5.48 # Mev
    energy_list = []

    energy_error_list = []

    stopping_power_list = []

    def __init__(self, data_file: str, pressure: int, end_point: int = None):
        """read the data of given csv file and put it in a dataframe

        Args:
            data_file: name of the csv file to be read
            end_point: end point of the data. Everything after this value in mV will be deleted from the dataframe
            pressure: the pressure of the gas during the measurement in mbar
        """
        # pressure to bar
        self.pressure = pressure/1000 
        Measurement.pressure_list.append(self.pressure)
        Measurement.path_length_list.append(self.pressure_to_path_length(self.pressure))


        self.data_file = data_file
        self.df_diagram = pd.read_csv(data_file)
        self.df_diagram = self.df_diagram.loc[(self.df_diagram['y0000'] > 0)]

        if end_point != None:
            self.df_diagram = self.df_diagram.loc[(self.df_diagram['x0000'] < end_point)]
        
        self.df_diagram['x0000'] /= 1000 # from mV to V
        self.df_diagram["error_counts"] = np.sqrt(self.df_diagram['y0000'])
        self.title, _ = self.data_file.split('.')
        
        
    def data_plot(self):
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
        # print(self.result_signal.fit_report())
        self.fit_mu = self.result_signal.params['mu'].value
        self.fit_mu_err = self.result_signal.params['mu'].stderr
        self.fit_gauss1mu = self.result_signal.params['gauss1_mu'].value
        self.fit_gauss1mu_err = self.result_signal.params['gauss1_mu'].stderr

        # only calculate the conversion factor in vacuum
        if self.pressure < 0.1:
            self.calc_conversion_factor()

        Measurement.volt.append(self.fit_gauss1mu)
        Measurement.volt_error.append(self.fit_gauss1mu_err)
        Measurement.energy_list.append(self.volt_to_energy(self.fit_gauss1mu))
        Measurement.energy_error_list.append(self.volt_to_energy(self.fit_gauss1mu_err))

    def exp_decay(x: float, a: float, b:float, c: float) -> float:
            """Exponential decay function to fit for the (energy, path length) diagram

            Args:
                x: The path length
                a: a constant times the exponential
                b: decay factor
                c: extra translation of the function

            Returns:
                Energy of the alpha particle with a given path length
            """            
            energy = c - a * b**x
            return energy

    @classmethod
    def energy_fit(cls):
        """fit of the energy diagram with the exponential decay function
        """        
        energy_weight = [1/energy_err for energy_err in cls.energy_error_list]
        cls.model_energy = models.Model(cls.exp_decay, nan_policy='propagate')
        cls.result_energy = cls.model_energy.fit(cls.energy_list, x=cls.path_length_list, weights=energy_weight, a=0.1, b=5, c=2)
        print(cls.result_energy.fit_report())
        cls.a = cls.result_energy.params['a'].value
        cls.b = cls.result_energy.params['b'].value
        cls.c = cls.result_energy.params['c'].value


    @classmethod
    def energy_plot(cls):
        """Plot a diagram of the energy of the alpha particle against the path length 
        """
        cls.energy_list = [cls.exp_decay(num, a=cls.a, b=cls.b, c=cls.c) for num in cls.path_length_list]
        cls.path_length_continuous = np.arange(cls.path_length_list[0], 4, 0.01)
        cls.energy_continuous = [cls.exp_decay(num, a=cls.a, b=cls.b, c=cls.c) if cls.exp_decay(num, a=cls.a, b=cls.b, c=cls.c) > 0 else 0 for num in cls.path_length_continuous]

        fig = plt.figure("(E,x)_diagram.png")
        plt.errorbar(cls.path_length_list, cls.energy_list, yerr=cls.energy_error_list, fmt='bo', ecolor='k', label='Measured data')
        plt.plot(cls.path_length_continuous, cls.energy_continuous, 'g', label=f'E(x) = c - a*b^x')
        plt.plot(cls.path_length_list, cls.result_energy.best_fit, 'r', label=f'Energy fit in measured range')
        plt.xlabel('path length (cm)')
        plt.ylabel('Energy (MeV)')
        plt.legend(loc='upper right')

        # go to fits folder to save fit
        os.chdir(os.path.join(python_file_path, "Fits"))
        plt.savefig("(E,x)_diagram.png")

        # go to csv folder to read csv
        os.chdir(os.path.join(python_file_path, "CSV data"))
        plt.show()
    
    def calc_conversion_factor(self):
        """calculate the conversion factor to go from a voltage to an energy
        """        
        Measurement.conversion_factor = Measurement.energy_alpha_init/self.fit_gauss1mu

    @staticmethod
    def volt_to_energy(voltage: float) -> float:
        """converts a voltage to an energy value

        Args:
            voltage: voltage to be converted

        Returns:
            energy value corresponding to the input voltage
        """        
        energy = Measurement.conversion_factor * voltage
        return energy
    
    @staticmethod
    def pressure_to_path_length(pressure: float) -> float:
        """converts a given pressure to a path length

        Args:
            pressure: pressure of the measurement in bar

        Returns:
            path length in cm of the measurement relative to a standard pressure and the physical distance to the detector
        """ 
        # physical distance to the detector in cm       
        standard_length = 3
        standard_pressure = 1 # in bar
        path_length = standard_length * np.cbrt(pressure/standard_pressure)
        
        pressure_error = 1
        path_length_error = pressure_error * standard_length / (3 * np.cbrt(standard_pressure * pressure ** 2))
        return path_length
    
    @classmethod
    def stopping_power_function(cls, x: float) -> float:
        """stopping power function, which is minus the derivative of the energy function with respect to the path length

        Args:
            x: The path length

        Returns:
            stopping power with a given path length
        """        
        # stopping power = - dE/dx
        stopping_power = cls.a* np.log(cls.b) * cls.b**x
        return stopping_power
    
    @classmethod
    def stopping_power_plot(cls):
        """Plots the stopping power against the path length and plots an (E,x) and a (S,x) diagram
        """
        cls.stopping_power_list = [cls.stopping_power_function(num) for num in cls.path_length_list]
        cls.path_length_continuous = np.arange(cls.path_length_list[0], 4, 0.01)    
        cls.stopping_power_continuous = [cls.stopping_power_function(num) if cls.exp_decay(num, a=cls.a, b=cls.b, c=cls.c) > 0 else 0 for num in cls.path_length_continuous]
        fig = plt.figure("(S,x)_diagram.png")
        plt.scatter(cls.path_length_list, cls.stopping_power_list, c='blue', label='Calculated stopping power of measured data')  
        plt.plot(cls.path_length_continuous, cls.stopping_power_continuous, 'g-', label=f'S(x) = dE/dx = ln(b)*a*b^x')
        plt.plot(cls.path_length_list, cls.stopping_power_list, c='red', label='Stopping power function')
        plt.xlabel("path length (cm)")
        plt.ylabel("Stopping power (MeV/cm)")
        plt.legend(loc='upper left')

        # go to fits folder to save fit
        os.chdir(os.path.join(python_file_path, "Fits"))
        plt.savefig("(S,x)_diagram.png")

        # go to csv folder to read csv
        os.chdir(os.path.join(python_file_path, "CSV data"))
        plt.show()
    
    @classmethod
    def alpha_range(cls) -> float:
        """calculate the range of an alpha particle at atmospheric pressure

        Returns:
            the range of the alpha particle
        """        
        # we could also use c - E0 instead of a
        range_alpha_ray = np.log(cls.c/(cls.a))/np.log(cls.b)
        return range_alpha_ray


def measurement_air():
    """runs the experiment with different pressures in air
    """

    meas1_air_low_list = [20, 99, 199, 299, 399, 499, 599, 699, 799, 899]
    meas1_air_high_list = [22, 114, 213, 314, 415, 515, 614, 714, 814, 971]

    meas1_vacuum = Measurement("alfa bron 21 mbar.csv", end_point=1000, pressure = 21)
    meas1_vacuum.data_fit(start_expmu=0.05, start_gauss1_mu=0.2)
    # meas1_vacuum.plot_fit()

    meas1_air_100mbar = Measurement("alfa bron lucht 100 mbar.csv", end_point=1000, pressure = 100)
    meas1_air_100mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.18)
    # meas1_air_100mbar.plot_fit()

    meas1_air_200mbar = Measurement("alfa bron lucht 200 mbar.csv", end_point=1000, pressure = 200)
    meas1_air_200mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.2)
    # meas1_air_200mbar.plot_fit()

    meas1_air_300mbar = Measurement("alfa bron lucht 300 mbar.csv", end_point=1000, pressure = 300)
    meas1_air_300mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.18)
    # meas1_air_300mbar.plot_fit()

    meas1_air_400mbar = Measurement("alfa bron lucht 400 mbar.csv", end_point=1000, pressure = 400)
    meas1_air_400mbar.data_fit(start_expmu=0.025, start_gauss1_mu=0.17)
    # meas1_air_400mbar.plot_fit()

    meas1_air_500mbar = Measurement("alfa bron lucht 500 mbar.csv", end_point=1000, pressure = 500)
    meas1_air_500mbar.data_fit(start_expmu=0.025, start_gauss1_mu=0.16)
    # meas1_air_500mbar.plot_fit()

    meas1_air_600mbar = Measurement("alfa bron lucht 600 mbar.csv", end_point=1000, pressure = 600)
    meas1_air_600mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.14)
    # meas1_air_600mbar.plot_fit()

    meas1_air_700mbar = Measurement("alfa bron lucht 700 mbar.csv", end_point=1000, pressure = 700)
    meas1_air_700mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # meas1_air_700mbar.plot_fit()

    meas1_air_800mbar = Measurement("alfa bron lucht 800 mbar.csv", end_point=1000, pressure = 800)
    meas1_air_800mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.12)
    # meas1_air_800mbar.plot_fit()

    meas1_air_900mbar = Measurement("alfa bron lucht 900 mbar.csv", end_point=1000, pressure = 900)
    meas1_air_900mbar.data_fit(start_expmu=0.022, start_gauss1_mu=0.12)
    # meas1_air_900mbar.plot_fit()

    Measurement.energy_fit()
    Measurement.energy_plot()
    Measurement.stopping_power_plot()
    print(f"The range of an alpha particle in air with atmosiferic pressure is {round(Measurement.alpha_range(), 2)} cm")

def measurement_argon():
    """runs the experiment with different pressures in argon
    """

    argon_low_list = [20, 98, 199, 299, 399, 499, 599, 699]
    argon_high_list = [22, 117, 215, 317, 417, 526, 627, 727]

    argon_vacuum = Measurement("alfa bron argon 21 mbar.csv", end_point=1000, pressure = 21)
    argon_vacuum.data_fit(start_expmu=0.05, start_gauss1_mu=0.2)
    # argon_vacuum.plot_fit()

    argon_100mbar = Measurement("alfa bron argon 100 mbar.csv", end_point=1000, pressure = 100)
    argon_100mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.18)
    # argon_100mbar.plot_fit()

    argon_200mbar = Measurement("alfa bron argon 200 mbar.csv", end_point=1000, pressure = 200)
    argon_200mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.2)
    # argon_200mbar.plot_fit()

    argon_300mbar = Measurement("alfa bron argon 300 mbar.csv", end_point=1000, pressure = 300)
    argon_300mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.18)
    # argon_300mbar.plot_fit()

    argon_400mbar = Measurement("alfa bron argon 400 mbar.csv", end_point=1000, pressure = 400)
    argon_400mbar.data_fit(start_expmu=0.025, start_gauss1_mu=0.17)
    # argon_400mbar.plot_fit()

    argon_500mbar = Measurement("alfa bron argon 500 mbar.csv", end_point=1000, pressure = 500)
    argon_500mbar.data_fit(start_expmu=0.025, start_gauss1_mu=0.16)
    # argon_500mbar.plot_fit()

    argon_600mbar = Measurement("alfa bron argon 600 mbar.csv", end_point=1000, pressure = 600)
    argon_600mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.14)
    # argon_600mbar.plot_fit()

    argon_700mbar = Measurement("alfa bron argon 700 mbar.csv", end_point=1000, pressure = 700)
    argon_700mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # argon_700mbar.plot_fit()

    Measurement.energy_fit()
    Measurement.energy_plot()
    Measurement.stopping_power_plot()
    print(f"The range of an alpha particle in argon with atmospheric pressure is {round(Measurement.alpha_range(), 2)} cm")

def measurement_helium():
    """runs the experiment with different pressures in helium
    """

    helium_low_list = [20, 98, 199, 299, 399, 499, 599, 699]
    helium_high_list = [22, 117, 215, 317, 417, 526, 627, 727]

    argon_vacuum = Measurement("alfa bron argon 21 mbar.csv", end_point=1000, pressure = 21)
    argon_vacuum.data_fit(start_expmu=0.05, start_gauss1_mu=0.2)
    # argon_vacuum.plot_fit()

    helium_50mbar = Measurement("alfa bron helium 50 mbar.csv", end_point=1000, pressure = 50)
    helium_50mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.18)
    # helium_50mbar.plot_fit()

    helium_100mbar = Measurement("alfa bron helium 100 mbar.csv", end_point=1000, pressure = 100)
    helium_100mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.2)
    # helium_100mbar.plot_fit()

    helium_150mbar = Measurement("alfa bron helium 150 mbar.csv", end_point=1000, pressure = 150)
    helium_150mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.18)
    # helium_150mbar.plot_fit()

    helium_200mbar = Measurement("alfa bron helium 200 mbar.csv", end_point=1000, pressure = 200)
    helium_200mbar.data_fit(start_expmu=0.025, start_gauss1_mu=0.17)
    # helium_200mbar.plot_fit()

    helium_250mbar = Measurement("alfa bron helium 250 mbar.csv", end_point=1000, pressure = 250)
    helium_250mbar.data_fit(start_expmu=0.025, start_gauss1_mu=0.16)
    # helium_250mbar.plot_fit()

    helium_300mbar = Measurement("alfa bron helium 300 mbar.csv", end_point=1000, pressure = 300)
    helium_300mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.14)
    # helium_300mbar.plot_fit()

    helium_350mbar = Measurement("alfa bron helium 350 mbar.csv", end_point=1000, pressure = 350)
    helium_350mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_350mbar.plot_fit()

    helium_400mbar = Measurement("alfa bron helium 400 mbar.csv", end_point=1000, pressure = 400)
    helium_400mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_400mbar.plot_fit()

    helium_450mbar = Measurement("alfa bron helium 450 mbar.csv", end_point=1000, pressure = 450)
    helium_450mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_450mbar.plot_fit()

    helium_500mbar = Measurement("alfa bron helium 500 mbar.csv", end_point=1000, pressure = 500)
    helium_500mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_500mbar.plot_fit()

    helium_550mbar = Measurement("alfa bron helium 550 mbar.csv", end_point=1000, pressure = 550)
    helium_550mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_550mbar.plot_fit()

    helium_600mbar = Measurement("alfa bron helium 600 mbar.csv", end_point=1000, pressure = 600)
    helium_600mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_600mbar.plot_fit()

    helium_650mbar = Measurement("alfa bron helium 650 mbar.csv", end_point=1000, pressure = 650)
    helium_650mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_650mbar.plot_fit()

    helium_700mbar = Measurement("alfa bron helium 700 mbar.csv", end_point=1000, pressure = 700)
    helium_700mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_700mbar.plot_fit()

    helium_750mbar = Measurement("alfa bron helium 750 mbar.csv", end_point=1000, pressure = 750)
    helium_750mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_750mbar.plot_fit()

    helium_800mbar = Measurement("alfa bron helium 800 mbar.csv", end_point=1000, pressure = 800)
    helium_800mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_800mbar.plot_fit()

    helium_850mbar = Measurement("alfa bron helium 850 mbar.csv", end_point=1000, pressure = 850)
    helium_850mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_850mbar.plot_fit()

    helium_900mbar = Measurement("alfa bron hrlium 900 mbar.csv", end_point=1000, pressure = 900)
    helium_900mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_900mbar.plot_fit()

    Measurement.energy_fit()
    Measurement.energy_plot()
    Measurement.stopping_power_plot()
    print(f"The range of an alpha particle in helium with atmospheric pressure is {round(Measurement.alpha_range(), 2)} cm")

def run():
    """Runs measurements with different gasses
    """    
    measurement_air()


if __name__ == "__main__":
    run()