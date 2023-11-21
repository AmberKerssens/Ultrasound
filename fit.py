# Amber Kerssens en Rowan
# experiment ultrasound
# practicum 2

#absorption coefficient
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# data lists
P0_r = np.array([97.1,89.7,83.0,98.6,97.2])
P0_t2_r_e = np.array([33.6,18.9,9.2,4.1,1.8])
st_dev_P0_r = [0.22,0.16,0.14,0.16,0.17]
st_dev_P0_t2_r_e = [0.14,0.26,0.16,0.25,0.13]

# Define the exponential function
def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

# Sample data
verhouding_array = P0_t2_r_e / P0_r
thickness_material = np.array([10.23,14.92,20.07,30.10,50.16])
print(type(verhouding_array))

# Fit the exponential function to the data
params, covariance = curve_fit(exponential, thickness_material, verhouding_array)

# Get the fitted parameters
a_fit, b_fit, c_fit = params

# Generate the fitted curve
xfit = np.linspace(min(thickness_material), max(thickness_material), 100)
yfit = exponential(xfit, a_fit, b_fit, c_fit)

# Plot the original data and the fitted curve
plt.scatter(thickness_material, verhouding_array, label='Data')
plt.plot(xfit, yfit, label=f'Fit: $y = {a_fit:.2f}e^({b_fit:.2f}x) + {c_fit:.2f}$')
plt.title("exponential fit absorption coëfficient")
plt.xlabel("Thickness material in mm")
plt.ylabel("fraction P0r and P0rt^2")
plt.legend()
plt.show()

