import tensorflow as tf
import os



def process_ds(img_path, mask_path):   
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255

    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=3)
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, [56, 56], method='nearest')

    return img, mask

def create_ds(dir_image, dir_mask):
    train_data_ds = tf.data.Dataset.list_files(dir_image+"\\*.jpg", shuffle=False)
    mask_data_ds = tf.data.Dataset.list_files(dir_mask+"\\*.png", shuffle=False)

    ds = tf.data.Dataset.zip((train_data_ds, mask_data_ds))
    ds = ds.shuffle(len(os.listdir(dir_image)), reshuffle_each_iteration=False)
    ds = ds.map(process_ds, num_parallel_calls=tf.data.AUTOTUNE)

    return ds
