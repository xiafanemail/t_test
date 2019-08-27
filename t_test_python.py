#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import stats
from scipy.stats import sem
from scipy.stats import t

from  plotly.offline import plot
import plotly.graph_objs as go
import plotly.figure_factory as ff

 
# function for calculating the t-test for two independent samples
def independent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# calculate standard errors
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
	sed = sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value ppf-Percent point function (inverse of cdf — percentiles).
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value cdf-Cumulative density function
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p
 


# In[2]:


# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100)
data2 = 5 * randn(100)


# In[3]:


data1


# In[4]:


data2


# In[5]:


# calculate the t test
alpha = 0.05
t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')


# In[6]:


twosample_results = stats.ttest_ind(data1, data2)
twosample_results


# In[7]:


matrix_twosample = [
    ['', 'Test Statistic', 'p-value'],
    ['Sample Data', twosample_results[0], twosample_results[1]]
]
matrix_twosample


# In[8]:


twosample_table = ff.create_table(matrix_twosample, index=True)
twosample_table


# In[13]:


plot(twosample_table, filename='twosample-table.html')


# In[14]:


x = np.linspace(-4, 4, 160)
trace1 = go.Scatter(
    x = x,
    y = data1,
    mode = 'lines+markers',
)
trace2 = go.Scatter(
    x = x,
    y = data2,
    mode = 'lines+markers',
)
data = [trace1, trace2]

plot(data, filename='normal-dists-plot.html')


# In[ ]:




