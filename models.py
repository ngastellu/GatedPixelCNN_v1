import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedConv2d_h(nn.Conv2d):
    '''
    add a mask to the regular Conv2D function for the h-stack only, so that it cannot learn from the pixel being predicted
    '''
    def __init__(self, mask_type, channels, *args, **kwargs):
        super(MaskedConv2d_h, self).__init__(*args, **kwargs)
        assert mask_type in {'A'}  # mask A is for the first convolutional layer only
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks

        if mask_type=='A':
            self.mask[:, :, :, -1] = 0  # rightmost pixel in the convolution is the 'present' pixel, which should be blocked in the first layer

    def forward(self, x):
        self.weight.data *= self.mask  # at each forward pass, apply the mask to all the filters (zero-out information about the future)
        return super(MaskedConv2d_h, self).forward(x)

def gated_activation(input):
    # implement gated activation from Conditional Generation with PixelCNN Encoders
    assert (input.shape[1] % 2) == 0
    a, b = torch.chunk(input, 2, 1) # split input into two equal parts - only works for even number of filters
    a = torch.tanh(a)
    b = torch.sigmoid(b)

    return torch.mul(a,b) # return element-wise (sigmoid-gated) product

class Activation(nn.Module):
    '''
    choice of activation function, note that gated takes 2f filters as input and 1f filters as output. This must be accounted in the model
    currently it it setup for gating, which converges better anyway
    '''
    def __init__(self, activation_func, *args, **kwargs):
        super().__init__()
        if activation_func == 'gated':
            self.activation = gated_activation
        elif activation_func == 'relu':
            self.activation = F.relu

    def forward(self, input):
        return self.activation(input)

class StackedConvolution(nn.Module):
    '''
    a single convolutional layer, broken into a vertical and horizontal stack, with residuals in the horizontal
    '''
    def __init__(self, f_in, f_out, padding, dilation, *args, **kwargs):
        super(StackedConvolution, self).__init__(*args, **kwargs)

        self.v_Conv2d = nn.Conv2d(f_in, 2 * f_out, (2, 3), 1, padding * (1,1), dilation, bias=True, padding_mode='zeros') # v_stack convolution
        self.v_to_h_fc = nn.Conv2d(2 * f_out, f_out, 1, bias=True) # 1x1 convolution from v to h
        self.h_Conv2d = nn.Conv2d(f_in, f_out, (1, 2), 1, padding * (0,1), dilation, bias=True, padding_mode='zeros') # h_stack convolution
        self.h_to_h = nn.Conv2d(f_out, f_out, 1, bias=True) # 1x1 convolution from h to residual
        self.activation = Activation('gated') # for ReLU, must change number of filters as gated approach halves filters on each application

    def forward(self, v_in, h_in):
        residue = h_in.clone() # residual track

        '''
        we adopt a pad & crop approach to the convolutions
        at each layer the v_stack must be unpadded by 1, and then aligned with the h_stack
        for an accessible explanation, see http://sergeiturukin.com/2017/02/24/gated-pixelcnn.html
        '''

        v_in = self.v_Conv2d(v_in)[:, :, :-1, :]  # remove extra padding
        v_out = self.activation(v_in) # activation in v_stack
        v_to_h = self.v_to_h_fc(v_in)[:,:,:-1,:] # align v stack to h
        h_in = self.h_Conv2d(h_in)[:, :, :, :-1]  # unpad by 1 on rhs
        h_out = self.activation(torch.cat((h_in, v_to_h), 1)) # activation in h_stack
        h_out = self.h_to_h(h_out) + residue # re-add residue

        return v_out, h_out


class PixelCNN(nn.Module):  # Dense or residual, gated, blocked, dilated PixelCNN with batchnorm
    def __init__(self, filters, initial_convolution_size, dilation, layers, out_maps, padding, channels):
        super(PixelCNN, self).__init__()
        # some vestigial constants from prior implementations (dilation, unpadding...)
        # to run with relu instead of gated activation, have to take out a bunch of factors of 2 (gated activation takes 2*f filters and returns 1*f)

        ### initialize constants
        self.padding = padding # will always be 1 in this implementation
        self.layers_per_block = layers # number of stacked convolutional layers after the initial
        self.blocks = 1 # will always be 1 in this implementation
        self.layers = int(self.layers_per_block * self.blocks)
        self.initial_pad = (initial_convolution_size - 1) // 2 # padding size in initial layer (should be 1)
        self.main_pad = 1 # padding in stacked convolutions after initial - will always be 1 in this implementation
        initial_filters = filters # number of filters for initial convolutional layer
        self.input_depth = 1 # number of channels - always 1 in this implementation

        f_in = (np.ones(self.layers + 1) * filters).astype(int) # number of filters going into each layer
        f_out = (np.ones(self.layers + 1) * filters).astype(int) # number of filters coming out of each layer
        self.dilation = (np.ones(self.layers) * dilation).astype(int) # dilation will always be 1 in this implementation
        self.unpad = np.zeros(self.layers + 1).astype(int) # since dilation is always 1 and kernels are 3x3 (after a fashion), this will always be 1
        ###
        self.activation = Activation('gated') # initialize choice of activation function

        # initial layer
        self.v_initial_convolution = nn.Conv2d(self.input_depth, 2 * initial_filters, (initial_convolution_size//2 + 1, initial_convolution_size), 1, padding * (self.initial_pad + 1, self.initial_pad), padding_mode='zeros', bias=True)
        self.v_to_h_initial = nn.Conv2d(2 * initial_filters, initial_filters,1)
        self.h_initial_convolution = MaskedConv2d_h('A', self.input_depth, self.input_depth, initial_filters, (1, initial_convolution_size//2 + 1), 1, padding * (0, self.initial_pad), padding_mode='zeros', bias=True)
        self.h_to_h_initial = nn.Conv2d(initial_filters, initial_filters, 1)

        # stack layers in blocks
        self.conv_layer = []
        for j in range(self.blocks):
            self.conv_layer.append([StackedConvolution(f_in[i + j * self.layers_per_block], f_out[i + j * self.layers_per_block], padding, self.dilation[i + j * self.layers_per_block]) for i in range(self.layers_per_block)]) # stacked convolution (no blind spot)
        for j in range(self.blocks):
            self.conv_layer[j] = nn.ModuleList(self.conv_layer[j])
        self.conv_layer = nn.ModuleList(self.conv_layer)

        # output fully connected (1x1 convolution) layers
        self.fc1 = nn.Conv2d(f_out[-1], 256, (1,1)) # final filters are 256, but can really be any even number >= 4
        self.fc2 = nn.Conv2d(256 // 2, out_maps * channels, 1) # gated activation cuts filters by 2

    def forward(self, input):
        # initial convolutional - initialize stacks
        v_data = self.v_initial_convolution(input)[:, :, :-(2 * self.initial_pad), :]  # self.initial_pad will always be 1
        v_to_h_data = self.v_to_h_initial(v_data)[:,:,:-1,:] # align with h-stack
        h_data = self.h_initial_convolution(input)[:,:,:,:-self.initial_pad] # unpad rhs of image
        h_data = self.activation(torch.cat((v_to_h_data, h_data), dim=1))
        h_data = self.h_to_h_initial(h_data)
        v_data = self.activation(v_data)

        # hidden layers
        for i in range(self.blocks): # loop over convolutional layers (only one block in this implementation)
            for j in range(self.layers_per_block):
               v_data, h_data = self.conv_layer[i][j](v_data, h_data) # stacked convolutions fix blind spot, hooray!


        # output convolutions
        x = self.activation(self.fc1(h_data))
        x = self.fc2(x)

        return x
