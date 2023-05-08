import matplotlib.pyplot as plt
import numpy as np
import logit_amplification
import diptest
from scipy.stats import kurtosis, skew, shapiro
import matplotlib.pyplot as plt

dip_stats = []
kurt_stats = []
normalcies = []
for i in range(100):
    logits = logit_amplification.construct_lorenz(delta=i, gamma=0.5, to_average=140)
    dip_stats.append(diptest.dipstat(logits))
    kurt_stats.append(kurtosis(logits))
    normalcies.append((shapiro(logits)[0]))


print(dip_stats)
print(kurt_stats)
print(normalcies)

plt.plot(normalcies)
plt.show()
