import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from os import mkdir
from os.path import exists
from transformers import RobertaTokenizer

from model_utils import *
from model_cvae import CVAETransformer, loss_function
from datasets import *

# Some hyper-parameters
num_layers = 4
d_model = 768
d_latent = 300
num_heads = 6
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 102

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size

data_path = '../reddit'

max_length = 32  # Maximum number of tokens
buffer_size = 900000
batch_size = 128
num_epochs = 50
learning_rate = 1e-4
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6
lambda_kl_ceil = 0.08
lambda_bow_ceil = 1.0

total_steps = num_epochs * math.ceil(buffer_size / batch_size)
kl_annealing_ratio_s = 0.00
kl_annealing_ratio_t = 0.25

checkpoint_path = 'checkpoints/cvae_000_025_50_lr_1e-4_roberta'
log_path = 'log/cvae_000_025_50_lr_1e-4_roberta.txt'


def main():
    if not exists('log'):
        mkdir('log')
    f = open(log_path, 'a', encoding = 'utf-8')

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        train_dataset, val_dataset = create_reddit_datasets(tokenizer, data_path, buffer_size, batch_size, max_length)
        train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = mirrored_strategy.experimental_distribute_dataset(val_dataset)

        # Define the model.
        cvae = CVAETransformer(num_layers, d_model, d_latent, num_heads, dff, hidden_act,
            dropout_rate, layer_norm_eps, max_position_embed, vocab_size)

        # Build the model and initialize word embeddings from RoBERTa
        build_cvae_model(cvae, max_length, vocab_size)
        cvae.embedder.load_weights('../weights/roberta2cvae_embedder.h5')
        print('Word embeddings initialized from RoBERTa.')

        # Define optimizer and metrics.
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
            epsilon = adam_epsilon)

        train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        train_dec_loss = tf.keras.metrics.Mean(name = 'train_dec_loss')
        train_kl_div = tf.keras.metrics.Mean(name = 'train_kl_div')
        train_bow_loss = tf.keras.metrics.Mean(name = 'train_bow_loss')

        valid_loss = tf.keras.metrics.Mean(name = 'valid_loss')
        valid_dec_loss = tf.keras.metrics.Mean(name = 'valid_dec_loss')
        valid_kl_div = tf.keras.metrics.Mean(name = 'valid_kl_div')
        valid_bow_loss = tf.keras.metrics.Mean(name = 'valid_bow_loss')

        # Define the checkpoint manager.
        ckpt = tf.train.Checkpoint(model = cvae, optimizer = optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

        # If a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
            f.write('Latest checkpoint restored!!\n')

        @tf.function
        def train_step(dist_inputs, lambda_kl, lambda_bow):
            def step_fn(inputs):
                inp, tar_inp, tar_real = inputs
                enc_inp_padding_mask, enc_tar_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

                with tf.GradientTape() as tape:
                    dec_logits, _, mu_r, logvar_r, mu_p, logvar_p, bow_logits = cvae(inp, tar_inp,
                        True, enc_inp_padding_mask, enc_tar_padding_mask, combined_mask, dec_padding_mask, lambda_bow)
                    loss_per_example, dec_loss_per_example, kl_div_per_example, bow_loss_per_example = loss_function(
                        tar_real, dec_logits, mu_r, logvar_r, mu_p, logvar_p, bow_logits, lambda_kl, lambda_bow)
                    loss = tf.reduce_sum(loss_per_example) * (1.0 / batch_size)
                    dec_loss = tf.reduce_sum(dec_loss_per_example) * (1.0 / batch_size)
                    kl_div = tf.reduce_sum(kl_div_per_example) * (1.0 / batch_size)
                    bow_loss = tf.reduce_sum(bow_loss_per_example) * (1.0 / batch_size)

                gradients = tape.gradient(loss, cvae.trainable_variables)
                optimizer.apply_gradients(zip(gradients, cvae.trainable_variables))
                return loss, dec_loss, kl_div, bow_loss

            loss_per_replica, dec_loss_per_replica, kl_div_per_replica, bow_loss_per_replica = mirrored_strategy.run(
                step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_per_replica, axis = None)
            mean_dec_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, dec_loss_per_replica, axis = None)
            mean_kl_div = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, kl_div_per_replica, axis = None)
            mean_bow_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, bow_loss_per_replica, axis = None)

            train_loss(mean_loss)
            train_dec_loss(mean_dec_loss)
            train_kl_div(mean_kl_div)
            train_bow_loss(mean_bow_loss)
            return mean_loss, mean_dec_loss, mean_kl_div, mean_bow_loss

        @tf.function
        def valid_step(dist_inputs):
            def step_fn(inputs):
                inp, tar_inp, tar_real = inputs
                enc_inp_padding_mask, enc_tar_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

                dec_logits, _, mu_r, logvar_r, mu_p, logvar_p, bow_logits = cvae(inp, tar_inp,
                    False, enc_inp_padding_mask, enc_tar_padding_mask, combined_mask, dec_padding_mask, lambda_bow_ceil)
                loss_per_example, dec_loss_per_example, kl_div_per_example, bow_loss_per_example = loss_function(
                    tar_real, dec_logits, mu_r, logvar_r, mu_p, logvar_p, bow_logits, lambda_kl_ceil, lambda_bow_ceil)
                loss = tf.reduce_sum(loss_per_example) * (1.0 / batch_size)
                dec_loss = tf.reduce_sum(dec_loss_per_example) * (1.0 / batch_size)
                kl_div = tf.reduce_sum(kl_div_per_example) * (1.0 / batch_size)
                bow_loss = tf.reduce_sum(bow_loss_per_example) * (1.0 / batch_size)

                return loss, dec_loss, kl_div, bow_loss

            loss_per_replica, dec_loss_per_replica, kl_div_per_replica, bow_loss_per_replica = mirrored_strategy.run(
                step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_per_replica, axis = None)
            mean_dec_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, dec_loss_per_replica, axis = None)
            mean_kl_div = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, kl_div_per_replica, axis = None)
            mean_bow_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, bow_loss_per_replica, axis = None)

            valid_loss(mean_loss)
            valid_dec_loss(mean_dec_loss)
            valid_kl_div(mean_kl_div)
            valid_bow_loss(mean_bow_loss)

        def get_lambda_kl(current_n_steps):
            return lambda_kl_ceil
            progress = current_n_steps / total_steps
            if tf.less_equal(progress, kl_annealing_ratio_s):
                return 0.0
            elif tf.greater_equal(progress, kl_annealing_ratio_t):
                return lambda_kl_ceil
            else:
                progress = (progress - kl_annealing_ratio_s) / (kl_annealing_ratio_t - kl_annealing_ratio_s)
                return (tf.tanh(6 * progress - 3) + 1) * 0.5 * lambda_kl_ceil

        def get_lambda_bow(current_n_steps):
            return lambda_bow_ceil
            progress = current_n_steps / total_steps
            if tf.less_equal(progress, kl_annealing_ratio_s):
                return 0.0
            else:
                return lambda_bow_ceil

        # Start training
        n_steps = tf.Variable(0.0, name = 'n_steps', trainable = False, dtype = tf.float32)
        for epoch in range(num_epochs):
            start = time.time()

            train_loss.reset_states()
            train_dec_loss.reset_states()
            train_kl_div.reset_states()
            train_bow_loss.reset_states()

            for (batch, inputs) in enumerate(train_dataset):
                n_steps.assign_add(1.0)
                lambda_kl = get_lambda_kl(n_steps)
                lambda_bow = get_lambda_bow(n_steps)

                loss, dec_loss, kl_div, bow_loss = train_step(inputs, lambda_kl, lambda_bow)
                mean_loss = train_loss.result()
                mean_dec_loss = train_dec_loss.result()
                mean_kl_div = train_kl_div.result()
                mean_bow_loss = train_bow_loss.result()

                print('Epoch {} batch {} mean_loss {:.4f} ({:.4f}) mean_dec_loss {:.4f} ({:.4f}) mean_kl_div {:.4f} * {:.4f} ({:.4f}) mean_bow_loss {:.4f} * {:.4f} ({:.4f})'.format(
                    epoch + 1, batch, mean_loss, loss, mean_dec_loss, dec_loss, lambda_kl, mean_kl_div, kl_div, lambda_bow, mean_bow_loss, bow_loss))
                f.write('Epoch {} batch {} mean_loss {:.4f} ({:.4f}) mean_dec_loss {:.4f} ({:.4f}) mean_kl_div {:.4f} * {:.4f} ({:.4f}) mean_bow_loss {:.4f} * {:.4f} ({:.4f})\n'.format(
                    epoch + 1, batch, mean_loss, loss, mean_dec_loss, dec_loss, lambda_kl, mean_kl_div, kl_div, lambda_bow, mean_bow_loss, bow_loss))

            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
            f.write('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

            epoch_loss = train_loss.result()
            epoch_dec_loss = train_dec_loss.result()
            epoch_kl_div = train_kl_div.result()
            epoch_bow_loss = train_bow_loss.result()
            print('Epoch {} loss {:.4f} dec_loss {:.4f} kl_div {:.4f} bow_loss {:.4f}'.format(epoch + 1,
                epoch_loss, epoch_dec_loss, epoch_kl_div, epoch_bow_loss))
            f.write('Epoch {} loss {:.4f} dec_loss {:.4f} kl_div {:.4f} bow_loss {:.4f}\n'.format(epoch + 1,
                epoch_loss, epoch_dec_loss, epoch_kl_div, epoch_bow_loss))

            current_time = time.time()
            print('Time taken for 1 epoch: {} secs'.format(current_time - start))
            f.write('Time taken for 1 epoch: {} secs\n'.format(current_time - start))

            valid_loss.reset_states()
            valid_dec_loss.reset_states()
            valid_kl_div.reset_states()
            valid_bow_loss.reset_states()

            for inputs in val_dataset:
                valid_step(inputs)
            epoch_val_loss = valid_loss.result()
            epoch_val_dec_loss = valid_dec_loss.result()
            epoch_val_kl_div = valid_kl_div.result()
            epoch_val_bow_loss = valid_bow_loss.result()
            print('Epoch {} valid loss {:.4f} dec_loss {:.4f} kl_div {:.4f} bow_loss {:.4f}\n'.format(epoch + 1,
                epoch_val_loss, epoch_val_dec_loss, epoch_val_kl_div, epoch_val_bow_loss))
            f.write('Epoch {} valid loss {:.4f} dec_loss {:.4f} kl_div {:.4f} bow_loss {:.4f}\n\n'.format(epoch + 1,
                epoch_val_loss, epoch_val_dec_loss, epoch_val_kl_div, epoch_val_bow_loss))

    f.close()

if __name__ == '__main__':
    main()
