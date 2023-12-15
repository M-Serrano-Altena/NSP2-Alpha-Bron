"""
Calculates the range of an alpha particle in different gasses and plots the respective (E,x) and (S,x) diagrams.

Documentation: <https://m-serrano-altena.github.io/NSP2-Alpha-Ray/>
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


class Measurement:

    path_length_list = []
    path_length_error = []

    volt = []
    volt_error = []

    conversion_factor = 1 # placeholder value
    energy_alpha_init = 5.48 # Mev

    standard_length = 3 # physical distance to the detector in cm 
    standard_pressure = 1.01325 # = 1 atm in bar

    energy_list = []
    energy_error_list = []
    stopping_power_list = []

    path_length_gas_continuous = []
    energy_gas_continuous = []
    stopping_power_gas_continuous = []

    path_length_gas_data = []
    energy_gas_data = []
    stopping_power_gas_data = []

    path_length_gas_error = []
    energy_gas_error = []


    def __init__(self, data_file: str, pressure_start: int, pressure_end: int, end_point: int = None):
        """Read the data of given csv file and put it in a dataframe and calculate the path length

        Args:
            data_file: Name of the csv file to be read
            end_point: End point of the data. Everything after this value in mV will be deleted from the dataframe
            pressure_start: The pressure of the gas at the beginning of a measurement in mbar
            pressure_end: The pressure of the gas at the end of a measurement in mbar
        """
        # 1/1000 --> pressure to bar
        self.pressure = self.pressure_mean(pressure_start, pressure_end) / 1000 
        self.pressure_error = self.pressure_err(pressure_start, pressure_end) / 1000
        Measurement.path_length_list.append(Measurement.pressure_to_path_length(self.pressure))
        Measurement.path_length_error.append(Measurement.path_length_err(pressure=self.pressure, pressure_error=self.pressure_error))

        # go to CSV files directory
        os.chdir(os.path.join(python_file_path, "CSV data"))
        self.data_file = data_file
        self.df_diagram = pd.read_csv(data_file)
        self.df_diagram = self.df_diagram.loc[(self.df_diagram['y0000'] > 0)]

        if end_point != None:
            self.df_diagram = self.df_diagram.loc[(self.df_diagram['x0000'] < end_point)]
        
        self.df_diagram['x0000'] /= 1000 # from mV to V
        self.df_diagram["error_counts"] = np.sqrt(self.df_diagram['y0000'])
        self.title, _ = self.data_file.split('.')
    
    @classmethod
    def clear(cls):
        """Clears lists to get empty lists for the next gas
        """        
        cls.path_length_list = []
        cls.path_length_error = []

        cls.volt = []
        cls.volt_error = []

        cls.energy_list = []
        cls.energy_error_list = []
        cls.stopping_power_list = []

    @staticmethod
    def pressure_mean(pressure_start: int, pressure_end: int) -> int:
        """Calculate the mean of the start and end pressure

        $$P_{mean} = \\frac {P_{start} + P_{end}}{2}$$

        Args:
            pressure_start: The pressure of the gas at the beginning of a measurement in mbar
            pressure_end: The pressure of the gas at the end of a measurement in mbar

        Returns:
            The mean of the start and end pressure in mbar
        """        
        pressure_mean = round(np.mean((pressure_start, pressure_end)))
        return int(pressure_mean)

    @staticmethod
    def pressure_err(pressure_start: int, pressure_end: int) -> int:
        """Calculates the error of the mean pressure

        $$P_{err} = P_{end} - P_{mean}$$

        Args:
            pressure_start: The pressure of the gas at the beginning of a measurement in mbar
            pressure_end: The pressure of the gas at the end of a measurement in mbar

        Returns:
            The error of the mean pressure in mbar
        """        
        pressure_error = abs(round(pressure_end - np.mean((pressure_start, pressure_end))))
        return int(pressure_error)

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
        """Fits the measured data with an exponential gauss with an added normal gauss

        Args:
            start_expmu: Start mean value of the exponential gauss for the fit
            start_gauss1_mu: Start mean value of the normal gauss for the fit
        """        
        def exp_gauss(x: list, amplitude: float, mu: float, sigma: float, labda: float, gauss1_amplitude: float, gauss1_mu: float, gauss1_sigma: float) -> np.ndarray:
            """Exponential gauss with an added normal gauss

            Args:
                x: List of x coordinates
                amplitude: The amplitude of the exponential gauss
                mu: The mean value of the exponential gauss
                sigma: The standard devitaion of the exponential gauss. Always greater than zero
                labda: The rate of the exponential component. Always greater than zero
                gauss1_amplitude: The amplitude of the normal gauss
                gauss1_mu: The mean value of the normal gauss
                gauss1_sigma: The standard deviation of the normal gauss

            Returns:
                Exponential gauss with an added normal gauss
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
        if self.pressure < 0.05:
            self.calc_conversion_factor()

        Measurement.volt.append(self.fit_gauss1mu)
        Measurement.volt_error.append(self.fit_gauss1mu_err)
        Measurement.energy_list.append(self.volt_to_energy(self.fit_gauss1mu))
        Measurement.energy_error_list.append(self.volt_to_energy(self.fit_gauss1mu_err))

    def exp_decay(x: float, a: float, b:float, c: float) -> float:
            """Exponential decay function to fit for the (energy, path length) diagram

            $$E(x) = c - a b^x$$

            Args:
                x: The path length
                a: A constant times the exponential
                b: Decay factor
                c: Extra translation of the function

            Returns:
                Energy of the alpha particle for a given path length
            """            
            energy = c - a * b**x
            return energy

    @classmethod
    def energy_fit(cls):
        """Fit of the energy diagram with the exponential decay function
        """        
        energy_weight = [1/energy_err for energy_err in cls.energy_error_list]
        cls.model_energy = models.Model(cls.exp_decay, nan_policy='propagate')
        cls.result_energy = cls.model_energy.fit(cls.energy_list, x=cls.path_length_list, weights=energy_weight, a=0.1, b=5, c=2)
        # print(cls.result_energy.fit_report())
        cls.a = cls.result_energy.params['a'].value
        cls.b = cls.result_energy.params['b'].value
        cls.c = cls.result_energy.params['c'].value
        cls.a_err = cls.result_energy.params['a'].stderr
        cls.b_err = cls.result_energy.params['b'].stderr
        cls.c_err = cls.result_energy.params['c'].stderr


    @classmethod
    def energy_plot(cls, gas: str):
        """Plot a diagram of the energy of the alpha particle against the path length for a certain gas

        Args:
            gas: Name of the gas that is being used 
        """
  
        # cls.energy_list = [cls.exp_decay(num, a=cls.a, b=cls.b, c=cls.c) for num in cls.path_length_list]
        cls.path_length_continuous = np.arange(0, 5, 0.01)
        cls.energy_continuous = [cls.exp_decay(num, a=cls.a, b=cls.b, c=cls.c) if cls.exp_decay(num, a=cls.a, b=cls.b, c=cls.c) > 0 else 0 for num in cls.path_length_continuous]

        fig = plt.figure(f"(E,x)_diagram_{gas}.png")
        plt.errorbar(cls.path_length_list, cls.energy_list, xerr=cls.path_length_error, yerr=cls.energy_error_list, fmt='bo', ecolor='k', label='Measured data')
        plt.plot(cls.path_length_continuous, cls.energy_continuous, 'g', label=f'E(x) = c - a*b^x')
        plt.plot(cls.path_length_list, cls.result_energy.best_fit, 'r', label=f'Energy fit in measured range')
        plt.xlim(0,4.5)
        plt.ylim(0,6.5)
        plt.xlabel('Path length (cm)')
        plt.ylabel('Energy (MeV)')
        plt.legend(loc='lower left')

        # go to fits folder to save fit
        os.chdir(os.path.join(python_file_path, "Fits"))
        plt.savefig(f"(E,x)_diagram_{gas}.png")

        plt.show()
    
    def calc_conversion_factor(self):
        """Calculate the conversion factor to go from a voltage to an energy. Is only run with a vacuum measurement

        $$ a = \\frac{E_0}{V} $$

        where $E_0$ is the initial energy of the alpha particle, which equals 5.48 MeV, and V is the voltage at vacuum.
        """        
        Measurement.conversion_factor = Measurement.energy_alpha_init/self.fit_gauss1mu

    @staticmethod
    def volt_to_energy(voltage: float) -> float:
        """Converts a voltage to an energy value

        $$ E = a V, $$

        where $a$ is the conversion factor

        Args:
            voltage: Voltage measured by the detector

        Returns:
            Energy value corresponding to the input voltage
        """        
        energy = Measurement.conversion_factor * voltage
        return energy
    
    @classmethod
    def pressure_to_path_length(cls, pressure: float) -> float:
        """Converts a given pressure to a path length

        $$x = x_0 \sqrt[3]{\\frac{P}{P_0}}$$

        Args:
            pressure: Pressure of the measurement in bar

        Returns:
            Path length in cm of the measurement relative to a standard pressure and the physical distance to the detector
        """ 

        path_length = cls.standard_length * np.cbrt(pressure/cls.standard_pressure)
        return path_length
    
    @classmethod
    def path_length_err(cls, pressure: float, pressure_error: float) -> float:
        """Calculates the error of the path length based on the pressure error

        $$\Delta x = \\frac{x_0 \Delta P}{3\sqrt[3]{P^2 P_0}}$$

        Args:
            pressure: Mean pressure of the measurement in bar
            pressure_error: Error of the pressure in bar

        Returns:
            Error of the path length in bar
        """        
        path_length_error = pressure_error * cls.standard_length / (3 * np.cbrt(cls.standard_pressure * pressure**2))
        return path_length_error
    
    @classmethod
    def stopping_power_function(cls, x: float) -> float:
        """Stopping power function, which is the negative derivative of the energy function with respect to the path length.

        $$S(x) = - \\frac{dE}{dx} = a \ln{(b)} b^x,$$

        where a and b are calculated from the energy fit

        Args:
            x: The path length

        Returns:
            Stopping power with a given path length
        """        
        # stopping power = - dE/dx
        stopping_power = cls.a* np.log(cls.b) * cls.b**x
        return stopping_power
    
    @classmethod
    def stopping_power_err(cls, x: float, x_err: float) -> float:
        """gives the error on the stopping power by using the error propagation formula

        $$\Delta S = \sqrt{\\left(b^x \ln{(b)} \Delta a\\right)^2 + \\left(a x b^{x-1} \ln{(b)} + a b^{x-1})\Delta b \\right)^2 + \\left(a \ln{(b)}^2 b^x \Delta x \\right)^2}$$

        Args:
            x: the path length
            x_err: the error on the path length

        Returns:
            the error on the stopping power at a certain path length
        """       
        stopping_power_error = np.sqrt((cls.b**x * np.log(cls.b) * cls.a_err)**2 + ((cls.a * x * cls.b**(x-1) * np.log(cls.b) + cls.a * cls.b**(x-1))*cls.b_err)**2 + (cls.a * np.log(cls.b)**2 * cls.b**x * x_err)**2)
        return stopping_power_error
    
    @classmethod
    def range_err(cls) -> float:
        """gives the error on the range of an alpha particle in a certain gas

        $$\Delta d = \sqrt{\\left(\\frac{\Delta c}{c \ln{(b)}} \\right)^2 + \\left(\\frac{\Delta a}{a \ln{(b)}} \\right)^2 + \\left(\\frac{\Delta \ln({\\frac{c}{a})} }{b \ln{(b)}^2}\\right)^2}$$

        Returns:
            the error on the range in a certain gas
        """        
        range_error = np.sqrt((1/(cls.c * np.log(cls.b)) * cls.c_err)**2 + (1/(cls.a * np.log(cls.b)) * cls.a_err)**2 + (np.log(cls.c/cls.a)/(cls.b * np.log(cls.b)**2 ) * cls.b_err)**2)
        return range_error

    @classmethod
    def stopping_power_plot(cls, gas: str):
        """Plots the stopping power against the path length for a certain gas

        Args:
            gas: Name of the gas that is being used
        """
        cls.stopping_power_list = [cls.stopping_power_function(num) for num in cls.path_length_list]
        cls.stopping_power_error = [cls.stopping_power_err(x, x_err) for x, x_err in zip(cls.path_length_list, cls.path_length_error)]
        cls.path_length_continuous = np.arange(0, 5, 0.01)    
        cls.stopping_power_continuous = [cls.stopping_power_function(num) if cls.exp_decay(num, a=cls.a, b=cls.b, c=cls.c) > 0 else 0 for num in cls.path_length_continuous]
        
        fig = plt.figure(f"(S,x)_diagram_{gas}.png")
        plt.errorbar(cls.path_length_list, cls.stopping_power_list, xerr=cls.path_length_error, yerr=cls.stopping_power_error, fmt='bo', ecolor='k', label='Calculated stopping power of measured data')  
        plt.plot(cls.path_length_continuous, cls.stopping_power_continuous, 'g-', label=f'S(x) = dE/dx = ln(b)*a*b^x')
        plt.plot(cls.path_length_list, cls.stopping_power_list, c='red', label='Stopping power function')
        plt.xlim(0,4.5)
        plt.xlabel("Path length (cm)")
        plt.ylabel("Stopping power (MeV/cm)")
        plt.legend(loc='upper left')

        # go to fits folder to save fit
        os.chdir(os.path.join(python_file_path, "Fits"))
        plt.savefig(f"(S,x)_diagram_{gas}.png")

        plt.show()
    
    @classmethod
    def alpha_range(cls) -> float:
        """Calculate the range of an alpha particle at atmospheric pressure

        $$d = \\frac{\ln{\left(\\frac{c}{a}\\right)}}{\ln{(b)}}, $$

        where a, b and c are calculated by the energy fit.

        Returns:
            The range of the alpha particle
        """        
        # we could also use c - E0 instead of a
        range_alpha_ray = np.log(cls.c/(cls.a))/np.log(cls.b)
        return range_alpha_ray

    @classmethod
    def energy_plot_all(cls):
        """Plots the (energy, path length) diagram for all gasses
        """        
        fig = plt.figure(f"(E,x)_diagram_all.png")

        # plot continous functions
        plt.plot(cls.path_length_gas_continuous[0], cls.energy_gas_continuous[0], 'b-', label=f'Energy in air')
        plt.plot(cls.path_length_gas_continuous[1], cls.energy_gas_continuous[1], 'r-', label=f'Energy in argon')
        plt.plot(cls.path_length_gas_continuous[2], cls.energy_gas_continuous[2], 'g-', label=f'Energy in helium')

        # plot data points
        plt.errorbar(cls.path_length_gas_data[0], cls.energy_gas_data[0], xerr=cls.path_length_gas_error[0], yerr=cls.energy_gas_error[0], fmt='bo', ecolor='k', label='Measured energy')
        plt.errorbar(cls.path_length_gas_data[1], cls.energy_gas_data[1], xerr=cls.path_length_gas_error[1], yerr=cls.energy_gas_error[1], fmt='ro', ecolor='k')
        plt.errorbar(cls.path_length_gas_data[2], cls.energy_gas_data[2], xerr=cls.path_length_gas_error[2], yerr=cls.energy_gas_error[2], fmt='go', ecolor='k')

        # plot omitted data
        plt.errorbar(cls.path_length_omit, cls.energy_omit, xerr=cls.path_length_error_omit, yerr=cls.energy_error_omit, linestyle='None', marker='o', c='lime', ecolor='black', label='Measured data not used in fit')  

        plt.xlim(0, 4.5)
        plt.ylim(0, 6.5)
        plt.xlabel("Path length (cm)")
        plt.ylabel("Energy (MeV)")
        plt.legend(loc='lower left')

        # go to fits folder to save fit
        os.chdir(os.path.join(python_file_path, "Fits"))
        plt.savefig("(E,x)_diagram_all.png")

        plt.show()

    @classmethod
    def stopping_power_plot_all(cls):
        """Plots the (stopping power, path length) diagram for all gasses
        """        
        fig = plt.figure(f"(S,x)_diagram_all.png")

        # plot continuous functions
        plt.plot(cls.path_length_gas_continuous[0], cls.stopping_power_gas_continuous[0], 'b-', label=f'Stopping power in air')
        plt.plot(cls.path_length_gas_continuous[1], cls.stopping_power_gas_continuous[1], 'r-', label=f'Stopping power in argon')
        plt.plot(cls.path_length_gas_continuous[2], cls.stopping_power_gas_continuous[2], 'g-', label=f'Stopping power in helium')

        # plot data points
        plt.errorbar(cls.path_length_gas_data[0], cls.stopping_power_gas_data[0], xerr=cls.path_length_gas_error[0] , fmt='bo', ecolor='k', label='Calculated stopping power of measured data')
        plt.errorbar(cls.path_length_gas_data[1], cls.stopping_power_gas_data[1], xerr=cls.path_length_gas_error[1] , fmt='ro', ecolor='k')
        plt.errorbar(cls.path_length_gas_data[2], cls.stopping_power_gas_data[2], xerr=cls.path_length_gas_error[2] , fmt='go', ecolor='k')
      
        plt.xlim(0, 4.5)
        plt.ylim(0, 10)
        plt.xlabel("Path length (cm)")
        plt.ylabel("Stopping power (MeV/cm)")
        plt.legend(loc='upper left')

        # go to fits folder to save fit
        os.chdir(os.path.join(python_file_path, "Fits"))
        plt.savefig("(S,x)_diagram_all.png")

        plt.show()
    
    @classmethod
    def save_data(cls):
        """Saves the data of the different gasses
        """        
        cls.path_length_gas_continuous.append(cls.path_length_continuous)
        cls.energy_gas_continuous.append(cls.energy_continuous)
        cls.stopping_power_gas_continuous.append(cls.stopping_power_continuous)

        cls.path_length_gas_data.append(cls.path_length_list)
        cls.energy_gas_data.append(cls.energy_list)
        cls.stopping_power_gas_data.append(cls.stopping_power_list)

        cls.path_length_gas_error.append(cls.path_length_error)
        cls.energy_gas_error.append(cls.energy_error_list)


def measurement_air():
    """Runs the experiment with different pressures in air
    """

    air_start_list = [20, 99, 199, 299, 399, 499, 599, 699, 799, 899]
    air_end_list = [22, 114, 213, 314, 415, 515, 614, 714, 814, 971]

    meas1_vacuum = Measurement("alfa bron 21 mbar.csv", end_point=1000, pressure_start = air_start_list[0], pressure_end=air_end_list[0])
    meas1_vacuum.data_fit(start_expmu=0.05, start_gauss1_mu=0.2)
    # meas1_vacuum.data_plot()

    air_100mbar = Measurement("alfa bron lucht 100 mbar.csv", end_point=1000, pressure_start = air_start_list[1], pressure_end=air_end_list[1])
    air_100mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.18)
    # air_100mbar.data_plot()

    air_200mbar = Measurement("alfa bron lucht 200 mbar.csv", end_point=1000, pressure_start = air_start_list[2], pressure_end=air_end_list[2])
    air_200mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.2)
    # air_200mbar.data_plot()

    air_300mbar = Measurement("alfa bron lucht 300 mbar.csv", end_point=1000, pressure_start = air_start_list[3], pressure_end=air_end_list[3])
    air_300mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.18)
    # air_300mbar.data_plot()

    air_400mbar = Measurement("alfa bron lucht 400 mbar.csv", end_point=1000, pressure_start = air_start_list[4], pressure_end=air_end_list[4])
    air_400mbar.data_fit(start_expmu=0.025, start_gauss1_mu=0.17)
    # air_400mbar.data_plot()

    air_500mbar = Measurement("alfa bron lucht 500 mbar.csv", end_point=1000, pressure_start = air_start_list[5], pressure_end=air_end_list[5])
    air_500mbar.data_fit(start_expmu=0.025, start_gauss1_mu=0.16)
    # air_500mbar.data_plot()

    air_600mbar = Measurement("alfa bron lucht 600 mbar.csv", end_point=1000, pressure_start = air_start_list[6], pressure_end=air_end_list[6])
    air_600mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.14)
    # air_600mbar.data_plot()

    air_700mbar = Measurement("alfa bron lucht 700 mbar.csv", end_point=1000, pressure_start = air_start_list[7], pressure_end=air_end_list[7])
    air_700mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # air_700mbar.data_plot()

    air_800mbar = Measurement("alfa bron lucht 800 mbar.csv", end_point=1000, pressure_start = air_start_list[8], pressure_end=air_end_list[8])
    air_800mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.12)
    # air_800mbar.data_plot()

    air_900mbar = Measurement("alfa bron lucht 900 mbar.csv", end_point=1000, pressure_start = air_start_list[9], pressure_end=air_end_list[9])
    air_900mbar.data_fit(start_expmu=0.022, start_gauss1_mu=0.12)
    # air_900mbar.data_plot()

    Measurement.energy_fit()
    Measurement.energy_plot(gas='air')
    Measurement.stopping_power_plot(gas='air')
    print(f"The range of an alpha particle in air with atmosiferic pressure is {round(Measurement.alpha_range(), 2)} ± {round(Measurement.range_err(), 2)} cm")

    Measurement.save_data()
    Measurement.clear()

def measurement_argon():
    """Runs the experiment with different pressures in argon
    """

    argon_start_list = [20, 98, 199, 299, 399, 499, 599, 699]
    argon_end_list = [22, 117, 215, 317, 417, 526, 627, 756]

    argon_vacuum = Measurement("meting 4- alfa bron argon 21 mbar.csv", end_point=1000, pressure_start = argon_start_list[0], pressure_end=argon_end_list[0])
    argon_vacuum.data_fit(start_expmu=0.01, start_gauss1_mu=0.15)

    # to calculate the conversion factor, but not use this data point in the energy fit
    Measurement.clear()

    argon_100mbar = Measurement("meting 4- alfa bron argon 100 mbar.csv", end_point=1000, pressure_start = argon_start_list[1], pressure_end=argon_end_list[1])
    argon_100mbar.data_fit(start_expmu=0.01, start_gauss1_mu=0.13)
    # argon_100mbar.data_plot()

    argon_200mbar = Measurement("meting 4- alfa bron argon 200 mbar.csv", end_point=1000, pressure_start = argon_start_list[2], pressure_end=argon_end_list[2])
    argon_200mbar.data_fit(start_expmu=0.01, start_gauss1_mu=0.13)
    # argon_200mbar.data_plot()

    argon_300mbar = Measurement("meting 4- alfa bron argon 300 mbar.csv", end_point=1000, pressure_start = argon_start_list[3], pressure_end=argon_end_list[3])
    argon_300mbar.data_fit(start_expmu=0.06, start_gauss1_mu=0.12)
    # argon_300mbar.data_plot()

    argon_400mbar = Measurement("meting 4- alfa bron argon 400 mbar.csv", end_point=1000, pressure_start = argon_start_list[4], pressure_end=argon_end_list[4])
    argon_400mbar.data_fit(start_expmu=0.06, start_gauss1_mu=0.11)
    # argon_400mbar.data_plot()

    argon_500mbar = Measurement("meting 4- alfa bron argon 500 mbar.csv", end_point=1000, pressure_start = argon_start_list[5], pressure_end=argon_end_list[5])
    argon_500mbar.data_fit(start_expmu=0.01, start_gauss1_mu=0.09)
    # argon_500mbar.data_plot()

    argon_600mbar = Measurement("meting 4- alfa bron argon 600 mbar.csv", end_point=1000, pressure_start = argon_start_list[6], pressure_end=argon_end_list[6])
    argon_600mbar.data_fit(start_expmu=0.01, start_gauss1_mu=0.07)
    # argon_600mbar.data_plot()

    argon_700mbar = Measurement("meting 4- alfa bron argon 700 mbar.csv", end_point=1000, pressure_start = argon_start_list[7], pressure_end=argon_end_list[7])
    argon_700mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # argon_700mbar.data_plot()

    Measurement.energy_fit()
    Measurement.energy_plot(gas='argon')
    Measurement.stopping_power_plot(gas='argon')
    print(f"The range of an alpha particle in argon with atmospheric pressure is {round(Measurement.alpha_range(), 2)} ± {round(Measurement.range_err(), 2)} cm")

    Measurement.save_data()
    Measurement.clear()

def measurement_helium():
    """Runs the experiment with different pressures in helium
    """

    helium_start_list = [20, 49, 99, 149, 199, 249, 299, 349, 399, 449, 499, 549, 599, 649, 699]
    helium_end_list = [22, 64, 115, 166, 215, 267, 315, 367, 417, 470, 521, 581, 627, 677, 737]

    argon_vacuum = Measurement("meting 4- alfa bron argon 21 mbar.csv", end_point=1000, pressure_start = helium_start_list[0], pressure_end=helium_end_list[0])
    argon_vacuum.data_fit(start_expmu=0.01, start_gauss1_mu=0.15)
    # argon_vacuum.data_plot()

    helium_50mbar = Measurement("meting 5- alfa bron helium 50 mbar.csv", end_point=1000, pressure_start = helium_start_list[1], pressure_end=helium_end_list[1])
    helium_50mbar.data_fit(start_expmu=0.01, start_gauss1_mu=0.15)
    # helium_50mbar.data_plot()

    # to show the two omited data points independantly and not use them in the energy fit
    Measurement.energy_omit = Measurement.energy_list
    Measurement.path_length_omit = Measurement.path_length_list
    Measurement.energy_error_omit = Measurement.energy_error_list
    Measurement.path_length_error_omit = Measurement.path_length_error

    Measurement.clear()

    helium_100mbar = Measurement("meting 5- alfa bron helium 100 mbar.csv", end_point=1000, pressure_start = helium_start_list[2], pressure_end=helium_end_list[2])
    helium_100mbar.data_fit(start_expmu=0.01, start_gauss1_mu=0.15)
    # helium_100mbar.data_plot()

    helium_150mbar = Measurement("meting 5- alfa bron helium 150 mbar.csv", end_point=1000, pressure_start = helium_start_list[3], pressure_end=helium_end_list[3])
    helium_150mbar.data_fit(start_expmu=0.005, start_gauss1_mu=0.15)
    # helium_150mbar.data_plot()

    helium_200mbar = Measurement("meting 5- alfa bron helium 200 mbar.csv", end_point=1000, pressure_start = helium_start_list[4], pressure_end=helium_end_list[4])
    helium_200mbar.data_fit(start_expmu=0.005, start_gauss1_mu=0.15)
    # helium_200mbar.data_plot()

    helium_250mbar = Measurement("meting 5- alfa bron helium 250 mbar.csv", end_point=1000, pressure_start = helium_start_list[5], pressure_end=helium_end_list[5])
    helium_250mbar.data_fit(start_expmu=0.01, start_gauss1_mu=0.15)
    # helium_250mbar.data_plot()

    helium_300mbar = Measurement("meting 5- alfa bron helium 300 mbar.csv", end_point=1000, pressure_start = helium_start_list[6], pressure_end=helium_end_list[6])
    helium_300mbar.data_fit(start_expmu=0.005, start_gauss1_mu=0.15)
    # helium_300mbar.data_plot()

    helium_350mbar = Measurement("meting 5- alfa bron helium 350 mbar.csv", end_point=1000, pressure_start = helium_start_list[7], pressure_end=helium_end_list[7])
    helium_350mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_350mbar.data_plot()

    helium_400mbar = Measurement("meting 5- alfa bron helium 400 mbar.csv", end_point=1000, pressure_start = helium_start_list[8], pressure_end=helium_end_list[8])
    helium_400mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_400mbar.data_plot()

    helium_450mbar = Measurement("meting 5- alfa bron helium 450 mbar.csv", end_point=1000, pressure_start = helium_start_list[9], pressure_end=helium_end_list[9])
    helium_450mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_450mbar.data_plot()

    helium_500mbar = Measurement("meting 5- alfa bron helium 500 mbar.csv", end_point=1000, pressure_start = helium_start_list[10], pressure_end=helium_end_list[10])
    helium_500mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_500mbar.data_plot()

    helium_550mbar = Measurement("meting 5- alfa bron helium 550 mbar.csv", end_point=1000, pressure_start = helium_start_list[11], pressure_end=helium_end_list[11])
    helium_550mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_550mbar.data_plot()

    helium_600mbar = Measurement("meting 5- alfa bron helium 600 mbar.csv", end_point=1000, pressure_start = helium_start_list[12], pressure_end=helium_end_list[12])
    helium_600mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_600mbar.data_plot()

    helium_650mbar = Measurement("meting 5- alfa bron helium 650 mbar.csv", end_point=1000, pressure_start = helium_start_list[13], pressure_end=helium_end_list[13])
    helium_650mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_650mbar.data_plot()

    helium_700mbar = Measurement("meting 5- alfa bron helium 700 mbar.csv", end_point=1000, pressure_start = helium_start_list[14], pressure_end=helium_end_list[14])
    helium_700mbar.data_fit(start_expmu=0.05, start_gauss1_mu=0.13)
    # helium_700mbar.data_plot()

    Measurement.energy_fit()
    Measurement.energy_plot(gas='helium')
    Measurement.stopping_power_plot(gas='helium')
    print(f"The range of an alpha particle in helium with atmospheric pressure is {round(Measurement.alpha_range(), 2)} ± {round(Measurement.range_err(), 2)} cm")

    Measurement.save_data()
    Measurement.clear()

def run():
    """Runs measurements with different gasses and makes a combined (E,x) and (S,x) plot for all gasses.
    """    
    measurement_air()
    measurement_argon()
    measurement_helium()
    
    Measurement.energy_plot_all()
    Measurement.stopping_power_plot_all()


if __name__ == "__main__":
    run()