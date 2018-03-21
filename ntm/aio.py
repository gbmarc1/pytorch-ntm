"""All in one NTM. Encapsulation of all components."""
import torch
from torch import nn
from ntm.Variable import Variable

from .ntm import NTM
from .controller import LSTMController, FFWController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory
import torch.nn.functional as F


class EncapsulatedNTM(nn.Module):

    def __init__(self, num_inputs, num_outputs,
                 controller_size, controller_layers, num_heads, N, M, controller_type):
        """Initialize an EncapsulatedNTM.

        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """
        super(EncapsulatedNTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M

        # Create the NTM components
        memory = NTMMemory(N, M)

        CONTROLLERS = {'NTM-LSTM': LSTMController, 'NTM-FFW': FFWController}
        controller = CONTROLLERS[controller_type](num_inputs + M*num_heads, controller_size, controller_layers)

        heads = nn.ModuleList([])
        for i in range(num_heads):
            heads += [
                NTMReadHead(memory, controller_size),
                NTMWriteHead(memory, controller_size)
            ]

        self.ntm = NTM(num_inputs, num_outputs, controller, memory, heads)
        self.memory = memory

    def enable_cuda(self):
        self.cuda()
        self.ntm.controller.cuda()
        self.ntm.cuda()
        for head in self.ntm.heads:
            head.cuda()
        self.memory.cuda()

        print('Cuda Enabled!')

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.ntm.create_new_state(batch_size)

    def forward(self, x=None):
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.num_inputs))

        o, self.previous_state = self.ntm(x, self.previous_state)
        return o, self.previous_state


class VanillaLSTM(nn.Module):
    """Vanilla LSTM Network."""
    def __init__(self, num_inputs, num_outputs, controller_size, controller_layers):
        """Initialize VanillaLSTM.

        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        """
        super(VanillaLSTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_layers = controller_layers

        self.controller = LSTMController(num_inputs, controller_size, controller_layers)
        self.hidden2out = nn.Linear(controller_size, num_outputs)

    def enable_cuda(self):
        self.cuda()
        self.controller.cuda()
        print('Cuda Enabled!')

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.previous_state = self.controller.create_new_state(batch_size)

    def forward(self, x=None):
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.num_inputs))

        o, self.previous_state = self.controller(x, self.previous_state)
        return F.sigmoid(self.hidden2out(o)), self.previous_state

