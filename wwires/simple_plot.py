import matplotlib.pyplot as plt
import numpy as np

# plots original versus reconstructed. Takes channel to be plotted
# data must be of shape [n_samples, n_channels]
def diff_plot(data_orig, data_rec, ch, batch_num):
    plt.figure(1)
    plt.plot(data_orig[:, ch], label='Original')
    plt.plot(data_rec[:, ch], label='Reconstructed', marker='*', linestyle='None')
    plt.legend()
    plt.title('Batch ' + str(batch_num))
    plt.show()

def mask_plot(n_wires):
    n_wires=16
    array = [16, 32]
    mask = np.zeros((array[0], array[1]))
    for current_wire in range(n_wires):
        lin_mask = np.arange(current_wire, mask.size, n_wires)
        # x_mask = lin_mask[current_wire:n_channels//n_wires]//array[1]
        # y_mask = lin_mask[current_wire:n_channels//n_wires]%array[1]
        x_mask = lin_mask % array[0]
        y_mask = lin_mask // array[0]
        mask[x_mask, y_mask] = (current_wire+1)*256/n_wires