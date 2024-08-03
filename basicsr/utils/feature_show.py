import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def feature_show(x, name, save_dir = "plt"):
    x = x.squeeze().cpu().numpy()
    x = np.mean(x,axis=0)
    
    f = np.fft.ifft2(x)
    fshift = np.fft.fftshift(f)
    x = np.log(np.abs(fshift)+6e-6)
 
    plt.figure()
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.imshow(x , cmap='jet')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, name), dpi=200)
    plt.close()
    
