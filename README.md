# Tensorflow2 + Vitis-AI 
This branch contains the code for tensorflow2 and Vitis-AI quantization

The vitis-ai.pptx explains how to set up the environment for vitis-ai 3.0 as well as vmware 17.0

There are 3 datasets data test are used for tarining and testing the model in CNN

Calib_data is used to quantize the trained tensorflow2 model in combination with the quantization code in test.py

Finally, the arch.json file fore the ultra96v2 dpu is provided and you can run the script in terminal provided inisde the test.py file to compile the quantized model
