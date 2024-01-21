import numpy as np
from scipy.constants import k
import matplotlib.pyplot as plt

kl_divergence = lambda p, q: np.sum(p * np.log(p / q))#what about when q is zero?

boltzmann_probability = lambda energy, partition_fuction, temperature: np.e**(-energy / (k * temperature)) / partition_fuction

boltzmann_partition = lambda energies, temperature: np.sum(np.exp(-energies / (k * temperature)))

E_tot = np.arange(0, 10**(-19) + 10**(-23), 10**(-23))

def boltzmann_function(lowest_energy, highest_energy, temperature):
    E_select = np.full(len(E_tot), np.nan)
    E_relevant = np.arange(lowest_energy, highest_energy, 10**(-23))
    E_tot_as_list = list(E_tot)
    E_select[E_tot_as_list.index(lowest_energy):E_tot_as_list.index(highest_energy)] = E_relevant
    partition = boltzmann_partition(E_relevant, temperature)
    return np.nan_to_num(np.where(E_select != np.nan, boltzmann_probability(E_select, partition, temperature), 0))

T_list = [100, 200, 300, 400]

p_list = [[]]
for T in T_list:
    p_list[0].append(boltzmann_function(0, 10**(-19), T))
p_list.append(['blue', 'red', 'green', 'yellow'])


for i, distribution in enumerate(p_list[0]):
    plt.subplot(2, 2, i+1)
    color = p_list[1][i]
    plt.plot(E_tot, p_list[0][i], c=color, label=r'T(K) = {}'.format(T_list[i]))
    plt.title('Boltzmann '+str(i+1)) 
    plt.xlabel('Energy (J)')
    plt.ylabel('Probability')
    plt.legend()
plt.show()

for row_index, p_prob in enumerate(p_list[0]):
    for column_index, q_prob in enumerate(p_list[0]):
        plot_index = (row_index + 1) * 4 - (3 - column_index) #subplot index nums 1-16
        new_plot = plt.subplot(4, 4, plot_index)
        new_plot.set_title('KL(P||Q) = %f' % kl_divergence(np.array(p_prob[:-1]), np.array(q_prob[:-1])))
        plt.plot(E_tot, p_prob, c=p_list[1][row_index], label=r'T = {}K'.format((row_index+1)*100))
        plt.plot(E_tot, q_prob, c=p_list[1][column_index], label=r'T = {}K'.format((column_index+1)*100))
        plt.xlabel('Energy (J)')
        plt.ylabel('Probability')
        new_plot.legend()
plt.suptitle('KLD between Boltzmann Distributions of different temperatures (row P, col Q)')
plt.show()
