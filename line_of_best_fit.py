# import packages
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# add legend
run = [6, 9, 12, 12, 15, 21, 24, 24, 27, 30, 36, 39, 45, 48, 57, 60]
rise = [12, 18, 30, 42, 48, 78, 90, 96, 96, 90, 84, 78, 66, 54, 36, 24]

# create scatterplot
plt.scatter(run, rise)

# polynomial fit with degree = 2
model = np.poly1d(np.polyfit(run, rise, 2))

# add fitted polynomial line to scatterplot
polyline = np.linspace(1, 60, 50)
plt.scatter(run, rise)
plt.plot(polyline, model(polyline))
plt.show()

print(model)