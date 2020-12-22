import tensorflow as tf
from model_basics import *


def loss_function(real, dec_logits, mu_r, logvar_r, mu_p, logvar_p, bow_logits,
                  lambda_kl, lambda_bow):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits = True, reduction = 'none')

    # To be consistent with RoBERTa, the padding index is set to 1.
    mask = tf.math.logical_not(tf.math.equal(real, 1))

    # Calculate the decoding loss.
    dec_loss = scce(real, dec_logits)
    mask = tf.cast(mask, dtype = dec_loss.dtype)
    dec_loss *= mask
    dec_loss = tf.reduce_sum(dec_loss, 1) / tf.reduce_sum(mask, 1)

    # Calculate the KL divergence.
    kl_div = 0.5 * tf.reduce_sum(logvar_p - logvar_r - 1
                                 + tf.exp(logvar_r - logvar_p)
                                 + (mu_p - mu_r) ** 2 / tf.exp(logvar_p), axis = 1)

    # Calculate the bag-of-words loss.
    bow_loss = scce(real, bow_logits) * mask
    bow_loss = tf.reduce_sum(bow_loss, 1) / tf.reduce_sum(mask, 1)

    loss = dec_loss + lambda_kl * kl_div + lambda_bow * bow_loss

    return loss, dec_loss, kl_div, bow_loss


class Embedder(tf.keras.Model):
    def __init__(self, d_model, dropout_rate, layer_norm_eps,
                 max_position_embed, vocab_size):
        super().__init__(name = 'embedder')

        self.padding_idx = 1

        self.word_embeddings = tf.keras.layers.Embedding(vocab_size, d_model, name = 'word_embed')
        self.pos_embeddings = tf.keras.layers.Embedding(max_position_embed, d_model, name = 'pos_embed')

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon = layer_norm_eps,
            name = 'layernorm_embed')
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name = 'dropout_embed')

    def call(self, x, training):
        # x.shape == (batch_size, seq_len)

        seq_len = tf.shape(x)[1]

        # Add word embedding and position embedding.
        pos = tf.range(self.padding_idx + 1, seq_len + self.padding_idx + 1)
        pos = tf.broadcast_to(pos, tf.shape(x))
        x = self.word_embeddings(x)  # (batch_size, seq_len, d_model)
        x += self.pos_embeddings(pos)

        x = self.layernorm(x)
        x = self.dropout(x, training = training)

        return x  # (batch_size, seq_len, d_model)

class Encoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, hidden_act,
                 dropout_rate, layer_norm_eps, name):
        super().__init__(name = name)
        self.num_layers = num_layers
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, hidden_act, dropout_rate, layer_norm_eps, i)
            for i in range(num_layers)
        ]

    def call(self, x, training, mask):
        # x.shape == (batch_size, input_seq_len, d_model)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)

# Pool the outputs of the encoders.
class Pooler(tf.keras.Model):
    def __init__(self, d_model, name):
        super().__init__(name = name)
        self.attention_v = tf.keras.layers.Dense(1, use_bias = False, name = 'attention_v')
        self.attention_layer = tf.keras.layers.Dense(d_model, activation = 'tanh', name = 'attention_layer')

    def call(self, x, mask):
        # x.shape == (batch_size, seq_len, d_model)
        # mask.shape == (batch_size, 1, 1, seq_len)

        # Compute the attention scores.
        projected = self.attention_layer(x)  # (batch_size, seq_len, d_model)
        logits = tf.squeeze(self.attention_v(projected), 2)  # (batch_size, seq_len)
        logits += (tf.squeeze(mask) * -1e9)  # Mask out the padding positions
        scores = tf.expand_dims(tf.nn.softmax(logits), 1)  # (batch_size, 1, seq_len)

        # x.shape == (batch_size, d_model)
        x = tf.squeeze(tf.matmul(scores, x), 1)

        return x

class PriorNetwork(tf.keras.Model):
    def __init__(self, dff, hidden_act, d_latent):
        super().__init__(name = 'prior_net')
        # Hidden layers
        self.hidden_layer = tf.keras.layers.Dense(dff // 2, activation = act_funcs[hidden_act],
            name = 'hidden_layer')
        self.hidden_layer_mu = tf.keras.layers.Dense(dff // 4, activation = act_funcs[hidden_act],
            name = 'hidden_layer_mu')
        self.hidden_layer_logvar = tf.keras.layers.Dense(dff // 4, activation = act_funcs[hidden_act],
            name = 'hidden_layer_logvar')

        # Output layers
        self.output_layer_mu = tf.keras.layers.Dense(d_latent, activation = tf.nn.tanh,
            name = 'output_layer_mu')
        self.output_layer_logvar = tf.keras.layers.Dense(d_latent, activation = tf.nn.tanh,
            name = 'output_layer_logvar')

    def call(self, inp):
        # inp.shape == (batch_size, d_model)
        h = self.hidden_layer(inp)

        h_mu = self.hidden_layer_mu(h)
        mu = self.output_layer_mu(h_mu)

        h_logvar = self.hidden_layer_logvar(h)
        logvar = self.output_layer_logvar(h_logvar)

        # Reparameterization
        z = mu + tf.exp(0.5 * logvar) * tf.random.normal(tf.shape(logvar))

        return z, mu, logvar

class RecognitionNetwork(tf.keras.Model):
    def __init__(self, dff, hidden_act, d_latent):
        super().__init__(name = 'recog_net')
        # Hidden layers
        self.hidden_layer = tf.keras.layers.Dense(dff, activation = act_funcs[hidden_act],
            name = 'hidden_layer')
        self.hidden_layer_mu = tf.keras.layers.Dense(dff // 2, activation = act_funcs[hidden_act],
            name = 'hidden_layer_mu')
        self.hidden_layer_logvar = tf.keras.layers.Dense(dff // 2, activation = act_funcs[hidden_act],
            name = 'hidden_layer_logvar')

        # Output layers
        self.output_layer_mu = tf.keras.layers.Dense(d_latent, activation = tf.nn.tanh,
            name = 'output_layer_mu')
        self.output_layer_logvar = tf.keras.layers.Dense(d_latent, activation = tf.nn.tanh,
            name = 'output_layer_logvar')

    def call(self, tar, inp):
        # tar.shape == inp.shape == (batch_size, d_model)
        x = tf.concat([tar, inp], axis = 1)

        h = self.hidden_layer(x)

        h_mu = self.hidden_layer_mu(h)
        mu = self.output_layer_mu(h_mu)

        h_logvar = self.hidden_layer_logvar(h)
        logvar = self.output_layer_logvar(h_logvar)

        # Reparameterization
        z = mu + tf.exp(0.5 * logvar) * tf.random.normal(tf.shape(logvar))

        return z, mu, logvar

class BowNetwork(tf.keras.Model):
    def __init__(self, dff, hidden_act, vocab_size):
        super().__init__(name = 'bow_net')
        self.hidden_layer = tf.keras.layers.Dense(dff, activation = act_funcs[hidden_act],
            name = 'hidden_layer')
        self.output_layer = tf.keras.layers.Dense(vocab_size, name = 'output_layer')

    def call(self, x, tar_seq_len):
        # x.shape == (batch_size, d_model + d_latent)
        h = self.hidden_layer(x)
        bow_logits = self.output_layer(h)
        bow_logits = tf.broadcast_to(tf.expand_dims(bow_logits, 1),
            shape = [bow_logits.shape[0], tar_seq_len, bow_logits.shape[1]])  # (batch_size, tar_seq_len, vocab_size)

        return bow_logits

class Decoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, hidden_act,
                 dropout_rate, layer_norm_eps):
        super().__init__(name = 'decoder')
        self.num_layers = num_layers
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, hidden_act, dropout_rate, layer_norm_eps, i)
            for i in range(num_layers)
        ]

    def call(self, x, enc_output, z, training, look_ahead_mask, padding_mask):
        # x.shape == (batch_size, target_seq_len, d_model)
        # z.shape == (batch_size, target_seq_len, d_latent)

        attention_weights = {}
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, z, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

# All history utterances are concatenated into one sequence.
class CVAETransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, d_latent, num_heads, dff, hidden_act, dropout_rate,
                 layer_norm_eps, max_position_embed, vocab_size):
        super().__init__(name = 'cvae_transformer')

        self.embedder = Embedder(d_model, dropout_rate, layer_norm_eps,
            max_position_embed, vocab_size)

        self.encoder_inp = Encoder(num_layers, d_model, num_heads, dff, hidden_act,
            dropout_rate, layer_norm_eps, 'encoder_inp')
        self.pooler_inp = Pooler(d_model, 'pooler_inp')

        self.encoder_tar = Encoder(num_layers, d_model, num_heads, dff, hidden_act,
            dropout_rate, layer_norm_eps, 'encoder_tar')
        self.pooler_tar = Pooler(d_model, 'pooler_tar')

        self.prior_net = PriorNetwork(dff, hidden_act, d_latent)
        self.recog_net = RecognitionNetwork(dff, hidden_act, d_latent)
        self.bow_net = BowNetwork(dff, hidden_act, vocab_size)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, hidden_act,
            dropout_rate, layer_norm_eps)

        self.final_layer = tf.keras.layers.Dense(vocab_size, name = 'final_layer')

    def call(self, inp, tar, training,
             enc_inp_padding_mask, enc_tar_padding_mask, look_ahead_mask, dec_padding_mask, switch):
        # inp_embed.shape == (batch_size, input_seq_len, d_model)
        # tar_embed.shape == (batch_size, target_seq_len, d_model)
        inp_embed = self.embedder(inp, training)
        tar_embed = self.embedder(tar, training)

        # enc_inp_output.shape == (batch_size, input_seq_len, d_model)
        # enc_tar_output.shape == (batch_size, target_seq_len, d_model)
        enc_inp_output = self.encoder_inp(inp_embed, training, enc_inp_padding_mask)
        enc_tar_output = self.encoder_tar(tar_embed, training, enc_tar_padding_mask)

        # enc_inp_output_pooled.shape == (batch_size, d_model)
        # enc_tar_output_pooled.shape == (batch_size, d_model)
        enc_inp_output_pooled = self.pooler_inp(enc_inp_output, enc_inp_padding_mask)
        enc_tar_output_pooled = self.pooler_tar(enc_tar_output, enc_tar_padding_mask)

        # z.shape == (batch_size, d_latent)
        _, mu_p, logvar_p = self.prior_net(enc_inp_output_pooled)
        z, mu_r, logvar_r = self.recog_net(enc_tar_output_pooled, enc_inp_output_pooled)

        # z_bc.shape == (batch_size, target_seq_len, d_latent)
        z_bc = tf.broadcast_to(tf.expand_dims(z, 1), shape = [z.shape[0], tar_embed.shape[1], z.shape[1]])

        # dec_output.shape == (batch_size, target_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar_embed,
            enc_inp_output, z_bc, training, look_ahead_mask, dec_padding_mask)

        # dec_logits.shape == (batch_size, target_seq_len, vocab_size)
        dec_logits = self.final_layer(dec_output)

        bow_inp = tf.concat([enc_inp_output_pooled, z], axis = 1)  # (batch_size, d_model + d_latent)
        bow_logits = self.bow_net(bow_inp, dec_logits.shape[1])  # (batch_size, target_seq_len, vocab_size)

        return dec_logits, attention_weights, mu_r, logvar_r, mu_p, logvar_p, bow_logits

    def encode(self, inp, training, enc_padding_mask):
        # inp_embed.shape == (batch_size, input_seq_len, d_model)
        inp_embed = self.embedder(inp, training)

        # enc_output.shape == (batch_size, input_seq_len, d_model)
        enc_output = self.encoder_inp(inp_embed, training, enc_padding_mask)

        # enc_output_pooled.shape == (batch_size, d_model)
        enc_output_pooled = self.pooler_inp(enc_output, enc_padding_mask)

        # z.shape == (batch_size, d_latent)
        z, mu, logvar = self.prior_net(enc_output_pooled)

        return enc_output, z, mu, logvar

    def decode(self, enc_output, z, tar, training, look_ahead_mask, dec_padding_mask):
        # tar_embed.shape == (batch_size, target_seq_len, d_model)
        tar_embed = self.embedder(tar, training)

        # z_bc.shape == (batch_size, target_seq_len, d_latent)
        z_bc = tf.broadcast_to(tf.expand_dims(z, 1), shape = [z.shape[0], tar_embed.shape[1], z.shape[1]])

        # dec_output.shape == (batch_size, target_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar_embed,
            enc_output, z_bc, training, look_ahead_mask, dec_padding_mask)

        # dec_logits.shape == (batch_size, target_seq_len, vocab_size)
        dec_logits = self.final_layer(dec_output)

        return dec_logits, attention_weights
