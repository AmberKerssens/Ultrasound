import numpy as np
import matplotlib.pyplot as plt
from lmfit import models

# Create synthetic data with some noise
# np.random.seed(42)
# x_data = np.linspace(0, 10, 100)
# y_data = 2 * np.exp(-0.5 * x_data) + 0.1 * np.random.normal(size=len(x_data))

# data lists
P0_r = np.array([97.1,89.7,83.0,98.6,97.2])
P0_t2_r_e = np.array([33.6,18.9,9.2,4.1,1.8])
st_dev_P0_r = [0.22,0.16,0.14,0.16,0.17]
st_dev_P0_t2_r_e = [0.14,0.26,0.16,0.25,0.13]


x_data =  np.array([10.23,14.92,20.07,30.10,50.16])
y_data = P0_t2_r_e / P0_r

#error, N= 10
for i in range (len(x_data)):
    xerr = np.std(i) / np.sqrt(10)


for i in range (len(y_data)):
    yerr = xerr = np.std(i) / np.sqrt(10)


# Define the exponential model
def exponential_model(x, amplitude, decay_rate, offset):
    return amplitude * np.exp(-decay_rate * x) + offset

# Create an lmfit Model
exp_model = models.Model(exponential_model, name = "Absorbtion fit")

# Set initial parameter values
# params = exp_model.make_params(amplitude=1, decay_rate=0.3, offset=0)

# Fit the model to the data
result = exp_model.fit(y_data,  x=x_data, amplitude = 1, decay_rate = 0.01, offset = 0)
result.plot()

# Print the fit results
print(result.fit_report())

# Plot the original data and the fitted curve
# plt.scatter(x_data, y_data, label='Data')
# plt.plot(x_data, result.best_fit, label='Best Fit', color='red')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Exponential Fit using lmfit')
plt.show()
