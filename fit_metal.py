import numpy as np
import matplotlib.pyplot as plt
from lmfit import models


### 2.25 Mhz 

# Data
P0_r = np.array([36.0,27.3,38.5,33.1,58.7,66.5])
P0_t2_r_e = np.array([9.5,6.9,9.4,5.6,9.0,6.3])
st_dev_P0_r = np.array([0.14,0.18,0.14,0.13,0.14,0.14])
st_dev_P0_t2_r_e = np.array([0.18,0.13,0.19,0.15,0.13,0.09])

x_data = np.array([6.99,9.87,30.04,40.02,50.03,75.00])
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

## 5 Mhz
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit import models
from lmfit import Model


### 2.25 Mhz 

# Data
P0_r = np.array([38.5,14.6,34.3,13.3,47.2,40.6])
P0_t2_r_e = np.array([8.2,3.3,9.1,2.7,4.6,8.9])
st_dev_P0_r = np.array([0.43,0.34,0.13,0.15,0.10,0.53])
st_dev_P0_t2_r_e = np.array([0.32,0.29,0.13,0.17,0.35,0.75])

x_data = np.array([6.99,9.87,30.04,40.02,50.03,75.00])
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