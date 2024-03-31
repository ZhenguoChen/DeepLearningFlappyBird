import tensorflow as tf


def convert_tf1_to_tf2(checkpoint_path, output_prefix):
    vars = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    dtypes = reader.get_variable_to_dtype_map()
    for key in dtypes.keys():
        vars[key] = tf.Variable(reader.get_tensor(key))
    return tf.train.Checkpoint(vars=vars).save(output_prefix)


print(convert_tf1_to_tf2("saved_networks", "tf2_weights"))
