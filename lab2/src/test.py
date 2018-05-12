import math
from lab2.src import help
import scipy.stats as stats
import scipy.optimize as opt
import scipy.special as special
import lab2.src.help as help
import numpy as np
import matplotlib.pyplot as plt

print(help.find_k(5600, 0.95, 0.8))

x = stats.binom.pmf(4, 5600, 0.95)
# plt.plot(x)

fig, ax = plt.subplots(1, 1)
n, p = 5600, 0.95
x = np.arange(stats.binom.ppf(0.01, n, p),
              stats.binom.ppf(0.99, n, p))
print(stats.binom.stats(n, p))
# ax.plot(x, stats.binom.cdf(x, n, p), 'b-', ms=8, label='binom pmf')
print(x)
# ax.vlines(x, 0, stats.binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)
# plt.show()
print(stats.binom.cdf(5300, n, p))
print(8 ** 2)

print("Дигамма 4: " + str(special.digamma(4)))
print("Дигамма 1000: " + str(special.digamma(1000)))
print("Гамма 4: " + str(math.gamma(4)))

arr = np.arange(1, 700000, step=1)
plt.close()
# plt.plot(arr, special.gamma(arr))
plt.plot(arr, special.digamma(arr), 'r')
plt.show()
print(np.log(math.e))

