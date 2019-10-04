import matplotlib.pyplot as plt
import math
import numpy as np
from time import time
plt.rcParams.update({'font.size': 16})
pi = math.pi


def c_n(n):
    coefficient = (2 / (pi * n)) * ((-1) ** n * (3 * pi - 4) + 4)
    return coefficient


def series_approx(coefficient_array, x_input):
    total = 0
    y_output = np.zeros(x_input.shape[0])

    for counter, x in enumerate(x_input):
        # print('counter is:', counter, '\nx is: ', x, )

        for i, coefficient in enumerate(coefficient_array):
            total += coefficient * math.sin((i + 1) * x)

        y_output[counter] = total
        total = 0  # rest total
    return y_output


coefficient_iterations = 4
comparing_coefficient_iterations = [coefficient_iterations] + [8, 50]
vector_coeff = np.vectorize(c_n)

x_domain = np.linspace(0, pi, 1000)  # iterated x_domain from 0 to pi
y_real = np.add(np.multiply(-3, x_domain), 4)
# print( 'X DOMAIN: ',x_domain)
# print(f'\n{coefficient_iterations} COEFFICIENTS: ', coefficients)

# print('Y-SERIES APPROX: ',y_estimates)


y_container = []
for value in comparing_coefficient_iterations:
    n = np.arange(1, value + 1, 1, dtype=np.int32)  # number of coefficients to generate
    coefficients = vector_coeff(n)  # C_n terms are generated

    y_estimates = series_approx(coefficients, x_domain)
    y_container.append(y_estimates)

fig, ax = plt.subplots(ncols=1, figsize=(12, 7))
ax.set_ylim((-2 * pi, 2 * pi))
ax.set_xlim((0, pi))
ax.plot(x_domain, y_real, '--', label='$y_1(x) = -3x + 4$')

for i, y in enumerate(y_container):
    ax.plot(x_domain, y, linewidth=2,
            label=f'Fourier Series Approximation: $y_1(x)$ with {comparing_coefficient_iterations[i]} terms')

ax.legend()
plt.show()

