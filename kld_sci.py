import numpy as np
from scipy.stats import norm, boltzmann
from matplotlib import pyplot as plt

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# comparing normal distributions

#setting possible values for the random variable x
x = np.linspace(-10, 10, 100)

standard_normal = norm.pdf(x, 0, 1) #the standard normal distribution
deviant_normal = norm.pdf(x, 0, 2) #a different standard deviation
different_normal = norm.pdf(x, 1, 1) #a different mean
other_normal = norm.pdf(x, 1, 2) #a different mean and standard deviation

#plot the four normal distributions
figure_1 = plt.figure(1)
figure_1.suptitle('The Normal Distributions')

standard_plot = plt.subplot(221)
standard_plot.set_title(r'$\mu = 0$, $\sigma = 1$')
plt.plot(x, standard_normal, c='red')

deviant_plot = plt.subplot(222)
deviant_plot.set_title(r'$\mu = 0$, $\sigma = 2$')
plt.plot(x, deviant_normal, c='blue')

different_plot = plt.subplot(223)
different_plot.set_title(r'$\mu = 1$, $\sigma = 1$')
plt.plot(x, different_normal, c='green')

other_plot = plt.subplot(224)
other_plot.set_title(r'$\mu = 1$, $\sigma = 2$')
plt.plot(x, other_normal, c='yellow')

plt.show()

#plot and calculate the KLD of every possible pair of these normal distributions
figure_2 = plt.figure(2)
figure_2.suptitle('A KLD pyplot Matrix of the Normal Distributions')

normal_list = [standard_normal, deviant_normal, different_normal, other_normal]

def pdf_color(pdf):
    if pdf is standard_normal:
        color = 'red'
    elif pdf is deviant_normal:
        color = 'blue'
    elif pdf is different_normal:
        color = 'green'
    elif pdf is other_normal:
        color = 'yellow'
    return color

def pdf_mean_std(pdf):
    if pdf is standard_normal:
        mean = 0
        std = 1
    elif pdf is deviant_normal:
        mean = 0
        std = 2
    elif pdf is different_normal:
        mean = 1
        std = 1
    elif pdf is other_normal:
        mean = 1
        std = 2
    return mean, std 

def remove_pdf(pdf_list, pdf):
    return [i for i in pdf_list if i is not pdf]

for row_index, p_pdf in enumerate(normal_list): #the row determines p, the 'true' distribution
    for column_index, q_pdf in enumerate(normal_list):
        plot_index = (row_index + 1) * 4 - (3 - column_index) #subplot index nums 1-16)
        new_plot = plt.subplot(4, 4, plot_index)
        new_plot.set_title('KL(P||Q) = %1.3f' % kl_divergence(p_pdf, q_pdf))
        p_mean, p_std = pdf_mean_std(p_pdf)
        q_mean, q_std = pdf_mean_std(q_pdf)
        plt.plot(x, p_pdf, c=pdf_color(p_pdf), label=r'$\mu_p = {}$, $\sigma_p = {}$'.format("%.2f" % p_mean, "%.2f" % p_std))
        plt.plot(x, q_pdf, c=pdf_color(q_pdf), label=r'$\mu_q = {}$, $\sigma_q = {}$'.format("%.2f" % q_mean, "%.2f" % q_std))
        new_plot.legend()
plt.show()
