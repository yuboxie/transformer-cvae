import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from os import mkdir
from os.path import exists, join

def create_reddit_datasets(tokenizer, path, buffer_size, batch_size, max_length):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))
    print('Reading data from \"{}\"...'.format(path))

    SOS_ID = tokenizer.encode('')[0]
    EOS_ID = tokenizer.encode('')[1]

    def create_dataset(dataset, N):
        inp_arr = np.ones((N, max_length), dtype = np.int32)
        tar_inp_arr = np.ones((N, max_length), dtype = np.int32)
        tar_real_arr = np.ones((N, max_length), dtype = np.int32)

        f_input = open(join(path, 'input_tokens_{}.txt'.format(dataset)), 'r')
        f_target = open(join(path, 'target_tokens_{}.txt'.format(dataset)), 'r')

        for i, line in tqdm(enumerate(f_input), total = N):
            input_ids = [SOS_ID] + [int(x) for x in line.strip().split(',')] + [EOS_ID]
            inp_arr[i,:len(input_ids)] = input_ids
        for i, line in tqdm(enumerate(f_target), total = N):
            target_ids = [SOS_ID] + [int(x) for x in line.strip().split(',')] + [EOS_ID]
            tar_inp_arr[i,:(len(target_ids)-1)] = target_ids[:-1]
            tar_real_arr[i,:(len(target_ids)-1)] = target_ids[1:]

        return (tf.data.Dataset.from_tensor_slices(inp_arr),
                tf.data.Dataset.from_tensor_slices(tar_inp_arr),
                tf.data.Dataset.from_tensor_slices(tar_real_arr))

    train_dataset = create_dataset('train', 900000)
    val_dataset = create_dataset('test', 100000)

    train_dataset = tf.data.Dataset.zip(train_dataset).shuffle(buffer_size).batch(batch_size)
    val_dataset = tf.data.Dataset.zip(val_dataset).batch(batch_size)

    return train_dataset, val_dataset
