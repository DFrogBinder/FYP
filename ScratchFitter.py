import numpy as np
from scipy.optimize import curve_fit


# Creating a function to model and create data
def func(x, a, b): 
    return  b*np.exp(a * (-x))
# Generating clean data
x = np.linspace(0, 10, 100)
y = func(x, 1, 2)
# Adding noise to the data
yn = y + 0.9 * np.random.normal(size=len(x))
# Executing curve_fit on noisy data
popt, pcov = curve_fit(func, x, yn)
# popt returns the best fit values for parameters of
# the given model (func).
print(popt)