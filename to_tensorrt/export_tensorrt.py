import torch
import torch_tensorrt
import os.path as osp
import cv2
from SMFANet import SMFANet

# Initialize the model and load weights
def initialize_model_and_load_weights():
    model = SMFANet()
    model.load_state_dict(torch.load('../pretrain/SMFANet_DF2K_100w_x4SR.pth')['params'])
    return model

# Perform tensorrt optimization and save the optimized model
def optimize_and_save_model(input_shape, model):
    model = model.eval().to("cuda")
    inputs = [torch.randn(input_shape).cuda()]

    optimized_model = torch_tensorrt.compile(
        model,
        inputs=inputs,
        ir="dynamo"
    )

    save_path = osp.join('tensorrt_model', f'{model.__class__.__name__}.ts')
    torch_tensorrt.save(optimized_model, save_path, output_format="torchscript", inputs=inputs)

    print(f"Model has been optimized and saved to {save_path}.")
    return  

if __name__ == '__main__':
    model = initialize_model_and_load_weights()

    input_img = cv2.imread('example_img/000.png').transpose((2, 0, 1))
    input_img = torch.from_numpy(input_img).unsqueeze(0) / 255.

    optimize_and_save_model(input_img.shape, model)

     
    