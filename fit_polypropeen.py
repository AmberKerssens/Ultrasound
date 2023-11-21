import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit import models
from lmfit import Model

# Data
P0_r = np.array([97.1, 89.7, 83.0, 98.6, 97.2])
P0_t2_r_e = np.array([33.6, 18.9, 9.2, 4.1, 1.8])
st_dev_P0_r = np.array([0.22, 0.16, 0.14, 0.16, 0.17])
st_dev_P0_t2_r_e = np.array([0.14, 0.26, 0.16, 0.25, 0.13])

x_data = np.array([10.23, 14.92, 20.07, 30.10, 50.16])
y_data = P0_t2_r_e / P0_r

# Calculate errors as the standard deviations
xerr = st_dev_P0_r / P0_r
yerr = np.sqrt((st_dev_P0_t2_r_e / P0_t2_r_e)**2 + (st_dev_P0_r / P0_r)**2) * (P0_t2_r_e / P0_r)

# # Define the exponential model
def exponential_model(x, amplitude, decay_rate, offset):
    return amplitude * np.exp(-decay_rate * x) + offset

# Create an lmfit Model
exp_model = models.Model(exponential_model, name="Absorption fit")
# exp_model = Model(exponential_model)

# Set initial parameter values
params = exp_model.make_params(amplitude=1, decay_rate=0.1, offset=0)

# Fit the model to the data with weights
result = exp_model.fit(y_data, params, x=x_data, weights=1.0 / yerr)

# Plot the original data and the fitted curve with error bars
# plt.errorbar(x_data, y_data, yerr=yerr, fmt='o', label='Data with Error Bars')
# plt.plot(x_data, result.best_fit, label='Best Fit', color='red')
# plt.legend()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Exponential Fit using lmfit with Error Bars')
# plt.show()

# Print the fit results
result.plot()
plt.show()
print(result.fit_report())
