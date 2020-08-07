# coding:utf-8

import sys
from datetime import date

from matplotlib.dates import DateFormatter
from matplotlib.dates import DayLocator
from matplotlib.dates import MonthLocator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

today = date.today()
start = (today.year - 1, today.month, today.day)

xx, yy = np.meshgrid(np.arange(-2, 2, 0.1),
                     np.arange(-2, 2, 0.1) )

x = y = np.linspace(-2, 2, 2000)

z = xx ** 2 + yy ** 2
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z)
plt.show()
