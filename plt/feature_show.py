import numpy as np
import matplotlib.pyplot as plt
import os

def feature_show(x, name, save_dir = "plt/psd", psd = True):
    x = x.squeeze().cpu().detach().numpy()
    x = np.mean(x,axis=0)
    
    if psd:
        f = np.fft.ifft2(x)
        fshift = np.fft.fftshift(f)
        x = np.log(np.abs(fshift) + 6e-6)
 
    plt.figure()
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.imshow(x , cmap='jet')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, name), dpi=200)
    plt.close()

    print("save at :", os.path.join(save_dir, name))
    
