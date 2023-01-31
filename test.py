#todo follow vitis-ai tf2 instructions and quantize the model

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize 

#todo load the dataset and set up calib_dataset
#test first before running rest of the code

'''
#to run docker ./docker_run.sh xilinx/vitis-ai-cpu:latest
#to compile vai_c_tensorflow2 -m "quantized_model.h5" -a "arch.json" -o compiled -n "pcb"
'''

img_height = 224
img_width = 224
batch_size = 32
calib_ds = tf.keras.preprocessing.image_dataset_from_directory(
	'calib_data',
	labels=None,
	label_mode=None,
	color_mode='rgb',
	batch_size=1,
	image_size=(img_height, img_width),
	shuffle=False
	#validation_split=0.1,
	#subset="training"
	)


#error in binary class due to sigmoid
'''
fashion_mnist = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
'''

model = tf.keras.models.load_model('model.h5') #make sure in correct directory

from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model)
quantized_model = quantizer.quantize_model(calib_dataset=calib_ds,
					    calib_steps=200,
					    calib_batch_size=1,
					    verbose=2,
					    add_shape_info=False
					    )
print("done1")
#save the quantized model					    
quantized_model.save('quantized_model.h5')	 
print("done2")

test_data = tf.keras.utils.image_dataset_from_directory(
  'data',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
quantized_model.compile(optimizer="adam",
              loss='SparseCategoricalCrossentropy',
              metrics = ['accuracy'])

quantized_model.evaluate(test_data)
