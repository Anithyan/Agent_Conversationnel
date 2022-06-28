import subprocess, os
import tensorflow as tf
autoencoder = tf.keras.models.load_model('./model_2')
encoder = tf.keras.models.load_model('./model_2_encoder')
decoder = tf.keras.models.load_model('./model_2_decoder')

model_name = 'model_2' # string used to define filename of saved model
decoder.save(model_name + '_decoder.hdf5', include_optimizer=True)

out_dir = "./interactive/"+model_name + '-decoder-js'
if not os.path.exists(out_dir): 
    os.makedirs(out_dir)

cmd = 'tensorflowjs_converter '
cmd += '--input_format keras_saved_model '
cmd += "./interactive/"+model_name + '_decoder.hdf5 '
cmd += out_dir
subprocess.check_output(cmd, shell=True)