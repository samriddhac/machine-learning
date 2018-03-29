# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 19:55:18 2018

@author: Samriddha.Chatterjee
"""
import scipy as sp
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

sp.info(sp.stats)

x_rand = sp.randn(10)
print(x_rand)

print('Mean = ',x_rand.mean(), ' Min= ', x_rand.min(), ' Max= ',x_rand.max(), ' Variance= ',
      x_rand.var(), ' Std= ',x_rand.std(), ' Median= ',sp.median(x_rand),
      ' Mode= ',stats.mode(x_rand))

n, min_max, mean, var, skew, kurt = stats.describe(x_rand)

print('n= ',n, ' min_max= ', min_max, ' mean= ', mean, ' var= ',var, ' skew= ',skew, ' kurt= ',kurt)

"""To create a “frozen” Gaussian or Normal distribution with mean = 3.5 
and standard deviation = 2.0"""

n = stats.norm(loc=3.5, scale=2.0)
"""To draw a random number from this distribution we can execute:"""
print(n.rvs())

"""For a normal distribution the keyword parameter loc defines the mean 
and the keyword parameter scale defines the standard deviation. 
For other distributions these will correspond to appropriate parameters of the 
distribution; the parameters needed by a distribution is specified 
in the docstring of the distribution"""

print(stats.norm.rvs(loc=3.5, scale=2.0))

"""For continuous variates the probability density function (PDF) is proportional 
to the probability of the variate being in a small interval about the given value. 
The probability of the variate being in a finite interval is the 
integral of the PDF over the interval."""

print(stats.norm.pdf([-1.0, 0.0, 1.0], loc=0.0, scale=1.0))

"""For discrete variates the probability mass function (PMF) gives the probability 
of the variate having a value x."""

x = range(11)
print(stats.binom.pmf(x, 10, 0.5))

def plot_binom_pmf(n=5, p=0.5):
    x = range(n+1)
    y = stats.binom.pmf(x, n, p)
    
    plt.plot(x, y, "o", color="red")
    plt.xticks(x)
    plt.title("Binomial Probablity Distribution")
    plt.xlabel("Variable")
    plt.ylabel("Probablity")
    plt.draw()
    

plot_binom_pmf(n=10, p=0.5)    

"""CDF (Cumulative Density Function) gives the probability that the variate has a value less than or equal to the 
given value."""

print(stats.norm.cdf([0.0, 0.5], loc=0.0, scale=1.0))

def plot_norm_cdf(mean = 0.0, std = 1.0):
    x = sp.linspace((mean-3*std), (mean+3*std), 50)
    print('x=',x)
    y = stats.norm.cdf(x, loc=mean, scale=std)
    plt.plot(x,y, color="red")
    plt.title("Normal Probability Distribution(CDF)")
    plt.xlabel("Random Variable")
    plt.ylabel("Probability")
    plt.draw()
    
    
plot_norm_cdf()

def plot_norm_pdf(mean = 0.0, std = 1.0):
    x = sp.linspace((mean-3*std), (mean+3*std), 50)
    print('x=',x)
    y = stats.norm.pdf(x, loc=mean, scale=std)
    plt.plot(x,y, color="red")
    plt.title("Normal Probability Distribution(PDF)")
    plt.xlabel("Random Variable")
    plt.ylabel("Probability")
    plt.draw()
    
    
plot_norm_pdf()


"""PPF(Percent Point Function) is the inverse of the CDF. That is, PPF gives the 
value of the variate for which the cumulative probability has the given value."""
print(stats.norm.ppf(0.69146246, loc=0.0, scale=1.0))

def plot_norm_ppf(mean = 0.0, std = 1.0):
    x = sp.linspace(0.0, 1.0, 1000)
    print('x=',x)
    y = stats.norm.ppf(x, loc=mean, scale=std)
    plt.plot(x,y, color="red")
    plt.title("Normal Probability Distribution(PPF)")
    plt.xlabel("Probability")
    plt.ylabel("Random Variable")
    plt.draw()
    
    
plot_norm_ppf()


"""Survival function gives the probability that the variate has a value 
greater than the given value; SF = 1 - CDF."""
print(stats.norm.sf([0.0, 0.5], loc=0.0, scale=1.0))

def plot_norm_sf(mean = 0.0, std = 1.0):
    x = sp.linspace((mean-3*std), (mean+3*std), 50)
    print('x=',x)
    y = stats.norm.sf(x, loc=mean, scale=std)
    plt.plot(x,y, color="red")
    plt.title("Normal Probability Distribution(SF)")
    plt.xlabel("Random Variable")
    plt.ylabel("Probability")
    plt.draw()
    
    
plot_norm_sf()

"""ISF is the inverse of the survival function. It gives the value of the variate 
for which the survival function has the given value."""
print(stats.norm.ppf(0.30853754, loc=0.0, scale=1.0))

def plot_norm_isf(mean = 0.0, std = 1.0):
    x = sp.linspace(0.0, 1.0, 1000)
    print('x=',x)
    y = stats.norm.isf(x, loc=mean, scale=std)
    plt.plot(x,y, color="red")
    plt.title("Normal Probability Distribution(PPF)")
    plt.xlabel("Probability")
    plt.ylabel("Random Variable")
    plt.draw()
    
    
plot_norm_isf()

"""A random value or several random values can be drawn from a 
distribution using the rvs method of the appropriate class."""

print(stats.norm.rvs(loc=0.0, scale=1.0, size=100))
print(stats.poisson.rvs(1.0, size=100))

def simulate_poisson():
    mu = 1.44
    sigma = sp.sqrt(mu)
    mu_plus_sigma = mu + sigma
    counts = stats.poisson.rvs(mu, size=100)
    bins = range(0, max(counts)+2)
    print(counts)
    print(bins)
    plt.hist(counts,bins=bins,align='left', histtype='step', color='black')
    x = range(0,10)
    prob = stats.poisson.pmf(x,mu)*100
    plt.plot(x, prob, "o", color="red")
    l = sp.linspace(0, 11, 100)
    s = interpolate.spline(x, prob, l)
    print(s)
    plt.plot(l, s, color='blue')
    plt.xlabel("Number of counts per 2 seconds")
    plt.ylabel("Number of occurrences (Poisson)")
    
    
#    xx = sp.searchsorted(l,mu_plus_sigma) - 1
#    v = ((s[xx+1] -  s[xx])/(l[xx+1]-l[xx])) * (mu_plus_sigma - l[xx])
#    v += s[xx]
#
#    ax = plt.gca()
#    # Reset axis range and ticks.
#    ax.axis([-0.5,10, 0, 40])
#    ax.set_xticks(range(1,10), minor=True)
#    ax.set_yticks(range(0,41,8))
#    ax.set_yticks(range(4,41,8), minor=True)
#
#    # Draw arrow and then place an opaque box with μ in it.
#    ax.annotate("", xy=(mu,29), xycoords="data", xytext=(mu, 13),
#                textcoords="data", arrowprops=dict(arrowstyle="->",
#                                                   connectionstyle="arc3"))
#    bbox_props = dict(boxstyle="round", fc="w", ec="w")
#    ax.text(mu, 21, r"$\mu$", va="center", ha="center",
#            size=15, bbox=bbox_props)
#
#    # Draw arrow and then place an opaque box with σ in it.
#    ax.annotate("", xy=(mu,v), xytext=(mu_plus_sigma,v),
#                arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
#    bbox_props = dict(boxstyle="round", fc="w", ec="w")
#    ax.text(mu+(sigma/2.0), v, r"$\sigma$", va="center", ha="center",
#            size=15, bbox=bbox_props)
    plt.draw()
    
    
simulate_poisson()    


"""Example: Suppose you wanted to answer the question, 
“What is the probability that a random sample of 20 families in 
Canada will have an average of 1.5 pets or fewer?” where the mean of 
the population is 0.8 and the standard deviation of the population is 1.2."""

population_mean = sample_dist_mean = 0.8
population_std = 1.2
sample_size =20

sample_dist_std = 1.2/sp.sqrt(sample_size)
plot_norm_pdf(sample_dist_mean, sample_dist_std)
prob = stats.norm.cdf(1.5, loc = sample_dist_mean, scale = sample_dist_std)
print(prob)















