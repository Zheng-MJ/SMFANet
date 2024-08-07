from SaliencyModel.utils import vis_saliency, vis_saliency_kde,grad_abs_norm,prepare_images, make_pil_grid
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad
from SaliencyModel.BackProp import attribution_objective, Path_gradient
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.BackProp import GaussianBlurPath
import argparse
import os
import cv2
import torch
import torchvision
from models.SMFANet import SMFANet

def PIL2Tensor(pil_image):
    return torchvision.transforms.functional.to_tensor(pil_image)

def main(args):

    # basic settings
    w, h = args.x, args.y
    scale = args.scale
    window_size = args.window_size
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load model
    model = SMFANet(dim=48, n_blocks=12, upscaling_factor=4)
    model_path = os.path.join('pretrain', 'SMFANet_plus_DF2K_100w_x4SR.pth')
    model.load_state_dict(torch.load(model_path)['params'], strict=True)

    img_lr, img_hr = prepare_images(args.img_dir, scale = args.scale)
    tensor_lr = PIL2Tensor(img_lr)[:3]  # * 255.0
 
    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    position_pil = cv2_to_pil(draw_img)
    position_pil.save(os.path.join(args.save_dir, f'rec_{model.__class__.__name__}.png'))

    sigma = 1.2
    fold = 50
    l = 9
    alpha = 0.5

    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective, gaus_blur_path_func, cuda=True)
    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=scale)
    saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy,zoomin=scale)

    blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    pil = make_pil_grid([position_pil, saliency_image_abs, blend_abs_and_input, blend_kde_and_input])

    pil.save(os.path.join(args.save_dir, f'{model.__class__.__name__}.png'))
    print('save at:', os.path.join(args.save_dir, f'{model.__class__.__name__}.png'))

    gini_idx = gini(abs_normed_grad_numpy)
    diffusion_idx = (1 - gini_idx) * 100
    print(f'{model.__class__.__name__} DI of this case is {diffusion_idx}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("LAM")
    parser.add_argument("--scale", default=4, type=int)
    parser.add_argument("--window_size", default=24, type=int)
    parser.add_argument("--x", default=240, type=int)
    parser.add_argument("--y", default=40, type=int)
    parser.add_argument("--img_dir", default = os.path.join('assets', 'img078.png'), type = str)
    parser.add_argument("--save_dir", default = os.path.join('plt', 'lam'), type = str)

    args = parser.parse_args()
    main(args)

