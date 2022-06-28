import subprocess, os
import tensorflow as tf

#os.chdir(os.path.join(os.getcwd(),'./interactive/'))
#autoencoder = tf.keras.models.load_model('./model_15')
print(os.getcwd())
model_name = 'model_2'
decoder = tf.keras.models.load_model('../model_2_decoder')

decoder.save(model_name + '-decoder.hdf5', include_optimizer=True)
out_dir = model_name + '-decoder-js'

if not os.path.exists(out_dir): os.makedirs(out_dir)

cmd = 'tensorflowjs_converter '
cmd += '--input_format keras_saved_model '
cmd += model_name + '-decoder.hdf5 '
cmd += out_dir
subprocess.check_output(cmd, shell=True)