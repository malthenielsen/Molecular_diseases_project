import numpy as np
from matplotlib import pyplot as plt
import sys
N = 100
cells = np.zeros((N,N))
cells[0, :] = 10

#  diff_up = np.diff(cells, axis = 0)
#  diff_dw = np.diff(np.flipud(cells), axis = 0)

#  mask = np.concatenate((diff_dw,np.zeros((1,N))), axis =0)
P = 0.2
link_bin = []
for i in range(N):
    link_bin.append(np.random.choice(N, np.random.randint(20,N), replace = False))
    #  link_bin.append(np.random.choice(N, 8, replace = False))

def prob_func(data_array, p):
    N_ = data_array.shape[0]
    prob_array = np.random.uniform(0,1,N_*N_).reshape(N_,N_)
    prob_array = np.where(prob_array < p, 0, 1)
    data_array = data_array*prob_array
    return data_array
final_pulse = []
for i in range(1000):
    if i%100 == 0:
        cells[0,:] = 10
    link_mask = np.zeros_like(cells)
    for j in range(N):
        if j != N-1:
            diff = cells[link_bin[j], j] - cells[link_bin[j], j+1]
            for val in range(len(diff)):
                if diff[val] > 0 and abs(diff[val]) == 10:
                    link_mask[link_bin[j][val], j+1] = 10
                elif diff[val] < 0 and abs(diff[val]) == 10:
                    link_mask[link_bin[j][val], j] = 10
        else:
            diff = cells[link_bin[j], j] - cells[link_bin[j], 0]
            for val in range(len(diff)):
                if diff[val] > 0 and abs(diff[val]) == 10:
                    link_mask[link_bin[j][val], 0] = 10
                elif diff[val] < 0 and abs(diff[val]) == 10:
                    link_mask[link_bin[j][val], j] = 10


    diff_up = np.abs(np.diff(cells, axis = 0))
    diff_dw = np.abs(np.diff(np.flipud(cells), axis = 0))

    mask_up = np.concatenate((np.zeros((1,N)), diff_up), axis =0)
    mask_up = prob_func(mask_up, P)
    mask_dw = np.flipud(np.concatenate((np.zeros((1,N)), diff_dw), axis =0))
    mask_dw = prob_func(mask_dw, P)
    link_mask = prob_func(link_mask, P)

    cells[cells > 0] -= .5
    #  print(mask_up)
    #  print('cells')
    #  print(mask_dw)

    cells[np.where(mask_up == 10)] = 10
    cells[np.where(mask_dw == 10)] = 10
    cells[np.where(link_mask == 10)] = 10
    final_pulse.append(np.mean(cells[-5:, :]))
    if np.mean(cells) == 0:
        print('problem')
        break
    #  print(cells)
    if i%50 == 0:
        plt.imshow(cells, cmap = 'binary')
        plt.show()

plt.plot(final_pulse)
plt.show()
