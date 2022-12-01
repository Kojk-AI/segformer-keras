import tensorflow as tf
import tensorflow_addons as tfa

import os
from datetime import datetime
import yaml

import model
import dataset
from config import Config

def run():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg = Config()

    dataset_len = len(os.listdir(cfg.data.train_image_dir))

    train_ds = dataset.create_ds(cfg.data.train_image_dir, cfg.data.train_mask_dir).batch(cfg.training.batch_size)
    val_ds = dataset.create_ds(cfg.data.val_image_dir, cfg.data.val_mask_dir).batch(cfg.training.batch_size)

    segformer = model.segformer(
        cfg.model.num_classes, 
        cfg.model.kernel_sizes, 
        cfg.model.strides, 
        cfg.model.emb_sizes, 
        cfg.model.reduction_ratios, 
        cfg.model.mlp_expansions, 
        cfg.model.num_heads, 
        cfg.model.depths, 
        cfg.model.decoder_channels, 
        cfg.model.scale_factors, 
        cfg.model.input_shape
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tfa.optimizers.RectifiedAdam(
        learning_rate=1e-4, 
        total_steps=dataset_len//cfg.training.batch_size, 
        warmup_proportion=0.1, 
        min_lr=1e-5,)

    segformer.compile(
        optimizer=optimizer, 
        loss=loss, 
        metrics=["accuracy"]
    )

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="ckpts\\{epoch:02d}-{val_loss:.2f}.ckpt",
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=False)

    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir="logs/{}".format(ts)
    )

    if cfg.training.resume_training == True:
        try:
            segformer.load_weights(cfg.training.weights_path)
        except Exception as e:
            print("Unable to load weights...")
            print(str(e))

    segformer.fit(train_ds, epochs=cfg.training.num_epochs, validation_data=val_ds, callbacks=[tb_cb, ckpt_cb])

    segformer.save_weights("weights/segformer_trained_{}.ckpt".format(ts))

if __name__ == "__main__":
   run()
