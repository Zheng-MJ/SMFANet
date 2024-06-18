import torch
import torchvision.models as models
import os.path as osp
import cv2
import numpy as np

# Read image and tensorrt model to perform inference, record inference time and gpu memory 
def inference_and_record(input_data, model_path):
    optimized_model = torch.jit.load(model_path).cuda()
    inputs = input_data.cuda() 
     
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    my_stream = torch.cuda.Stream()

    runtime = 0
    iter = 500
    with torch.no_grad():
        for _ in range(iter):
            with torch.cuda.stream(my_stream):
                start.record()
                outputs = optimized_model(inputs)
                my_stream.synchronize()
                end.record()
            
            runtime += start.elapsed_time(end)

    avg_time = runtime / iter 
    max_memory = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2

    print(f'TensorRT Inference AvgTime: {avg_time} ms')    
    print(f"TensorRT Inference Max Memory: {max_memory} [M]")
    return outputs

# Save the picture 
def save_inferenced_image(output, save_path):
    output_np = output.squeeze().cpu().numpy()
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    if output_np.ndim > 2:
        output_np = np.transpose(output_np, (1, 2, 0))
    cv2.imwrite(save_path, output_np)
    print(f'Image has been saved to {save_path}')

if __name__ == "__main__":
    
    input_img = cv2.imread('example_img/000.png').transpose((2, 0, 1))
    input_img = torch.from_numpy(input_img).unsqueeze(0) / 255.

    model_path = osp.join('tensorrt_model', 'SMFANet.ts')
    output = inference_and_record(input_img, model_path)
    save_inferenced_image(output, 'example_img/output.png')