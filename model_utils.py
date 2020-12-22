import numpy as np
import tensorflow as tf

# Masking
def create_padding_mask(seq):
    # To be consistent with RoBERTa, the padding index is set to 1.
    seq = tf.cast(tf.math.equal(seq, 1), tf.float32)

    # Add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_inp_padding_mask = create_padding_mask(inp)
    enc_tar_padding_mask = create_padding_mask(tar)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_inp_padding_mask, enc_tar_padding_mask, combined_mask, dec_padding_mask

# Model initialization
def build_cvae_model(model, max_length, vocab_size):
    inp = np.ones((1, max_length), dtype = np.int32)
    inp[0,:max_length//2] = np.random.randint(2, vocab_size, size = max_length//2)
    tar = np.ones((1, max_length), dtype = np.int32)
    tar[0,:max_length//2] = np.random.randint(2, vocab_size, size = max_length//2)

    inp = tf.constant(inp)
    tar = tf.constant(tar)

    enc_inp_padding_mask, enc_tar_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)
    model(inp, tar, True, enc_inp_padding_mask, enc_tar_padding_mask, look_ahead_mask, dec_padding_mask, 1.0)
