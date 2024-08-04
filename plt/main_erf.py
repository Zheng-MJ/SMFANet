import os.path
import torch
import torch.nn.functional as F
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from torch import optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from basicsr.archs.SMFANet_arch import SMFANet

def get_dataset(data_dir):
    path = []
    for imgname in os.listdir(data_dir):
        path.append(os.path.join(data_dir, imgname))
    return path

def img_read(path):
    print("read :", path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert img is not None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)

def get_input_grad(model, samples):
    outputs = model(samples)
    out_size = outputs.size()
    central_point = outputs[:, :, out_size[2] // 2, out_size[3] // 2].sum()
    grad = torch.autograd.grad(central_point, samples)

    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map

def run_erf(model, device, data_dir, save_dir):
    h,w = 96,96
    meter = torch.zeros(h,w)
    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)
    optimizer.zero_grad()

    dataset = get_dataset(data_dir)
    for i, img_lr in enumerate(dataset):

        img_lr = img_read(img_lr)

        img_lr = F.interpolate(img_lr , size=(h,w), mode='nearest')
        img_lr = img_lr.to(device)
        img_lr.requires_grad = True

        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, img_lr)

        if np.isnan(np.sum(contribution_scores)):
            print('got NAN, next image')
            continue
        else:
            meter += contribution_scores
            print(f"{i + 1}/{len(dataset)} meter.shape is {meter.shape}")

    data = meter.numpy()
    data = np.log10(data + 1)
    data = data / np.max(data)

    plt.figure(figsize = (10, 10.75), dpi=40)
    ax = sns.heatmap(data,
                xticklabels=False,
                yticklabels=False, cmap='RdYlGn',
                center=0, annot=False, ax=None, cbar=False, annot_kws={"size": 24}, fmt='.2f')

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('top', size='5%', pad='2%')
    plt.colorbar(ax.get_children()[0], cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, f"{model.__class__.__name__}_erf.png"), dpi=400)
    print('save at:', os.path.join(save_dir, f"{model.__class__.__name__}_erf.png"))

def main(args):
    # basic settings
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = SMFANet(dim=48, n_blocks=12, upscaling_factor= 4).to(device)
    model_path = os.path.join('pretrain', 'SMFANet_plus_DF2K_100w_x4SR.pth')
    model.load_state_dict(torch.load(model_path)['params'], strict=True)

    run_erf(model, device, args.data_dir, args.save_dir)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("ERF")
    # Prepare the test images in the folder path
    parser.add_argument("--data_dir", default = os.path.join('datasets', 'Benchmarks', 'Urban100', 'LR_bicubic', 'X4'), type = str)
    parser.add_argument("--save_dir", default = os.path.join('plt', 'erf'), type = str)
    args = parser.parse_args()
    main(args)
 
