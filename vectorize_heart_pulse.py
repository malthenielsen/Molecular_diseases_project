import numpy as np
from matplotlib import pyplot as plt
import sys
import time
import copy as cp
import matplotlib.cm as cm
import matplotlib.animation as animation


N = 200
iterations = 1000
cells = np.zeros((N,N))

P = 0.005
Pt = 0.55
alpha = 0.05


prob_array_LR = np.random.uniform(0,1, N*N).reshape(N,N)
prob_array_RL = np.random.uniform(0,1, N*N).reshape(N,N)

dys = np.random.uniform(0,1, N*N).reshape(N,N)

def prob_func(data_array, prob_array, p):
    prob_array = np.where(prob_array < p, 0, 1)
    data_array = data_array*prob_array
    return data_array

def dysfunctional(data_array, prob_array, p, alpha):
    prob_array = np.where(prob_array < p, 0, 1)
    #  print(prob_array)
    chance = np.random.uniform(0,1,N*N).reshape(N,N)
    chance = np.where(chance < alpha, 1,0)
    prob_array = prob_array + chance
    prob_array[prob_array > 0] = 1
    #  prob_array = np.where(prob_array ==1, 0,1)
    data_array = data_array*prob_array
    return data_array

final_pulse = []
history = np.zeros((N,N,iterations))

for i in range(iterations):
    tic = time.perf_counter()
    if i%100 == 0:
        cells[0,:] = 10
    LR_ghost = np.concatenate((cells[:,-1].reshape(N,1), cells), axis = 1)
    diff_LR = (np.diff(LR_ghost, axis = 1))
    cell_flip = np.fliplr(cells)
    RL_ghost = np.concatenate((cell_flip[:,-1].reshape(N,1), cell_flip), axis = 1)
    diff_RL = np.fliplr((np.diff(RL_ghost)))


    diff_up = (np.diff(cells, axis = 0))
    diff_dw = (np.diff(np.flipud(cells), axis = 0))

    mask_up = np.concatenate((np.zeros((1,N)), diff_up), axis =0)
    #  mask_up = prob_func_twice(mask_up, prob_array_UP, P, alpha)
    mask_dw = np.flipud(np.concatenate((np.zeros((1,N)), diff_dw), axis =0))
    #  mask_dw = prob_func_twice(mask_dw,prob_array_DW, P, alpha)

    mask_LR = prob_func(diff_LR, prob_array_LR, Pt)
    mask_RL = prob_func(diff_RL, prob_array_RL, Pt)

    cells[cells > 0] -= 1.3
    #  cells -= 1/8*cells
    cells[cells < 0] = 0

    cells[np.where(mask_up == -10)] = 10
    cells[np.where(mask_dw == -10)] = 10
    cells[np.where(mask_LR == -10)] = 10
    cells[np.where(mask_RL == -10)] = 10
    cells = dysfunctional(cells,dys, P, alpha) 


    final_pulse.append(np.mean(cells[-5:, :]))
    #  print(cells)
    history[:,:,i] = cells/8
    toc = time.perf_counter()
    #  print(cells)
    print('Time ', toc - tic)
    if np.mean(cells) == 0:
        print('problem')
        break
    #  print(cells)
    #  if i%50 == 0:
    #      plt.imshow(cells, cmap = 'binary')
    #      plt.show()

plt.plot(final_pulse)
plt.show()

def make_mp4movie(cmap):
    frames = [] # for storing the generated images
    fig = plt.figure()
    for t in range(iterations):
        frames.append([plt.imshow(cmap[:,:,t], cmap='gray', vmin=0, vmax=1, animated=True)])
    
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,repeat_delay=1000)
    #  ani.save('Fibrillation_movie.mp4')
    plt.show()



make_mp4movie(history)
