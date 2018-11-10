import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.integrate as sc

x = np.linspace(0.01,3,100)
dx = x[1]-x[0]

def lognormal (x,mu=0,sigma=1):
    return (1/(x*sigma*np.sqrt(2*np.pi)))* np.exp(-(np.log(x)-mu)**2/(2*sigma**2))

def exponential(x,l = 1.0):
    if np.any(x) < 0:
        return 0
    else:
        return l*np.exp(-l*x)

mu = 0.0
sigma = 0.2
beta = 0.02#expectation value
lam = 1/beta

fct_lognormal = lambda x :(1/(x*sigma*np.sqrt(2*np.pi)))* np.exp(-(np.log(x)-mu)**2/(2*sigma**2))
fct_out = lambda x: np.log(lognormal(x,mu,sigma))
fct_exp = lambda x:exponential(x,lam)

fx = lognormal(x, mu, sigma)
print np.sum(fx)*dx
print "Integral:", sc.quad(fct_lognormal, 0, np.inf)
print "Integral:", sc.quad(fct_exp, 0, np.inf)

var = np.random.lognormal(mu,sigma,10000)
var2 = np.random.exponential(scale=beta, size=10000)
var3 = np.exp(np.random.normal(mu,sigma,size=10000))
var4 = np.random.normal(loc=0, scale=2.0, size=10000)
var5 = 2*np.random.normal(loc = 0, scale=1.0,size=10000)
print var3


plt.figure(1)
plt.plot(x,fx)
plt.plot(x, exponential(x,lam), "r")

# plt.figure(3)
# plt.hist(var, bins=100)
#
# plt.figure(4)
# plt.hist(np.log(var), bins=100)

plt.figure(5)
plt.hist(var2, bins =100)
plt.figure(6)
plt.hist(var3, bins =100)

plt.figure(7)
plt.hist([var4, var5], bins=100)


# plt.figure(2)
# plt.plot(x,y)
plt.show()