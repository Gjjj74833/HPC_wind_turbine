
import numpy as np
import seaborn as sns
import binsreg

# Create random data
x = np.random.normal(size=1000)
y = x + np.random.normal(size=1000)

# Define the data as a pandas DataFrame
import pandas as pd
data = pd.DataFrame({'x': x, 'y': y})

# Create a binned regression plot
est = binsreg.binsreg(y, x, data=data, nbins=40, polyreg=1)
plot = est.bins_plot

# Save the figure
plot.save('binned_regression_plot.png')

