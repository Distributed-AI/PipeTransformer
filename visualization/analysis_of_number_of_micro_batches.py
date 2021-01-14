"""
Experimental results are record at:
https://wandb.ai/automl/pipe_and_ddp/reports/PipeTransformer-Experiments-and-Performance-Analysis--VmlldzozNzYzMTQ
Please check "Chunks of Micro-batches" section for original experiment data.

Please contact chaoyang.he@usc.edu if you have any question regarding the result.
"""
# https://matplotlib.org/1.2.1/examples/pylab_examples/histogram_demo.html
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

mu, sigma = 100, 15
x = mu + sigma*np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
# y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.show()
