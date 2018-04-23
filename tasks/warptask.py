"""Warping Task."""
import math

from attr import attrs, attrib, Factory, validators
import torch
from torch import nn
from ntm.Variable import Variable
from torch import optim
import numpy as np
import torch.nn.functional as F


from ntm.aio import EncapsulatedNTM, VanillaLSTM


# Generator of randomized test sequences
def dataloader(num_batches,
               batch_size,
               seq_width,
               min_len,
               max_len):
    """Generator of random sequences for the warp task.

    Creates random batches of "bits" sequences.

    All the sequences within each batch have the same length.
    The length is [`min_len`, `max_len`]

    :param num_batches: Total number of batches to generate.
    :param seq_width: The width of each item in the sequence.
    :param batch_size: Batch size.
    :param min_len: Sequence minimum length.
    :param max_len: Sequence maximum length.

    NOTE: The input width is `seq_width + 1`, the additional input
    contain the delimiter.
    """
    #assert seq_width == 1

    sequences_data = np.transpose(np.load('./data/warptask/data_uniform_10.npy'))
    target_data = np.transpose(np.load('./data/warptask/target_uniform_10.npy'))

    seq_len = sequences_data.shape[0]
    for epoch in range(num_batches):

        for batch_num, (seq, tar) in enumerate(zip(np.array_split(sequences_data, int(math.ceil(sequences_data.shape[1]/batch_size)), axis=1),
                                             np.array_split(target_data, int(math.ceil(target_data.shape[1] / batch_size)), axis=1))
                                             ):

            seq = np.reshape(seq, (seq_len, seq.shape[1], 1))
            seq = Variable(torch.from_numpy(seq))

            # The input includes an additional channel used for the delimiter
            inp = Variable(torch.zeros(seq_len + 1, seq.shape[1], 2))
            inp[:seq_len, :, :1] = seq
            inp[seq_len, :, 1] = 1.0 # delimiter in our control channel

            tar = np.reshape(tar, (seq_len, tar.shape[1], 1))
            outp = Variable(torch.from_numpy(tar))

            yield batch_num+1, inp.float(), outp.long()


@attrs
class WarpTaskParams(object):
    name = attrib(default="warp-task")
    controller_size = attrib(default=100, convert=int)
    controller_layers = attrib(default=1,convert=int)
    controller_type = attrib(default='NTM-LSTM', convert=str, validator=validators.in_(['NTM-LSTM', 'NTM-FFW', 'LSTM']))
    num_heads = attrib(default=1, convert=int)
    head_activation_type = attrib(default='softplus', convert=str, validator=validators.in_(['softplus', 'relu']))
    sequence_width = attrib(default=8, convert=int)
    sequence_min_len = attrib(default=1,convert=int)
    sequence_max_len = attrib(default=20, convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=20, convert=int)
    num_batches = attrib(default=50000, convert=int)
    batch_size = attrib(default=1, convert=int)
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)


#
# To create a network simply instantiate the `:class:CopyTaskModelTraining`,
# all the components will be wired with the default values.
# In case you'd like to change any of defaults, do the following:
#
# > params = CopyTaskParams(batch_size=4)
# > model = CopyTaskModelTraining(params=params)
#
# Then use `model.net`, `model.optimizer` and `model.criterion` to train the
# network. Call `model.train_batch` for training and `model.evaluate`
# for evaluating.
#
# You may skip this alltogether, and use `:class:CopyTaskNTM` directly.
#

@attrs
class WarpTaskModelTraining(object):
    params = attrib(default=Factory(WarpTaskParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):

        # Choose between Neural Turing Machine or Vanilla LSTM
        ENCAPSULATION = {'NTM-LSTM': EncapsulatedNTM, 'NTM-FFW': EncapsulatedNTM, 'LSTM':VanillaLSTM}
        encapsulation = ENCAPSULATION[self.params.controller_type]

        head_activation_type = {'relu': F.relu, 'softplus': F.softplus}

        # Arguments for Classical model
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        classic_args = (2, 10,
                        self.params.controller_size, self.params.controller_layers)

        # Arguments for Neural Turing Machine Model
        ntm_args = classic_args + (self.params.num_heads,
                                   self.params.memory_n, self.params.memory_m,
                                   self.params.controller_type,
                                   head_activation_type[self.params.head_activation_type])

        # Choice of Args
        ARGUMENTS = {EncapsulatedNTM: ntm_args, VanillaLSTM: classic_args}

        net = encapsulation(*ARGUMENTS[encapsulation])
        net.multi_target = True

        return net

    @dataloader.default
    def default_dataloader(self):
        return dataloader(self.params.num_batches, self.params.batch_size,
                          self.params.sequence_width,
                          self.params.sequence_min_len, self.params.sequence_max_len)

    @criterion.default
    def default_criterion(self):
        return nn.CrossEntropyLoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
