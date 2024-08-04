### Export TensorRT
- The following procedures are used to generate our TensorRT models.

### Requirements
> - PyTorch 2.3.1, cuda-121
> - torch_tensorrt 2.3.0
> - Platforms: Ubuntu 18.04, NVIDIA GeForce RTX 4090

### Install dependencies
```
pip install torch_tensorrt
```
### Use the export script
```
cd to_tensorrt
python export_tensorrt.py
```
### Use the inference script
```
cd to_tensorrt
python inference_tensorrt.py
```
