### Plotting Script
The following process is used for visualizing features and plotting charts.
 
 
### Feature Visualization
The script for feature visualization is located at [plt/feature_show.py](feature_show.py). 

To execute the feature visualization, run the following command:
```
python main_psd.py
```

To visualize the local attribution map (LAM), run the following command:
```
python main_lam.py 
```

To generate the effective receptive field (ERF), prepare the test images, and run the following command:
```
python main_erf.py --data_dir datasets/Benchmarks/Urban100/LR_bicubic/X4
```



### Chart Plotting
To generate a comprehensive performance comparison chart, execute the following command:
```
python main_complexity.py
python main_efficiency.py
```
Use the PDF editor to adjust text position in the chart.

To calculate the model complexity with the fvcore library, run the following command 
```
# install dependent package
pip install fvcore
# run the script
python main_flops.py 
```
FLOPs is measured corresponding to an HR image of the size 1280×720 pixels.
