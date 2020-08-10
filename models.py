import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedConv2d(nn.Conv2d):  # add a mask to the regular Conv2D function, so that it cannot learn about the future
    def __init__(self, mask_type, channels, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}  # mask A is for the first convolutional layer only, all deeper layers use mask B
        _, _, kH, kW = self.weight.size()  # get the size of the convolutional filter
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks

        # spatial masking - prevent information from neighbours
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0  # mask type B allows access to the 'present' pixel, mask A does not
        self.mask[:, :, kH // 2 + 1:] = 0

        if channels > 1:
            # channel masking - block information from nearby color channels - ONLY 2 CHANNELS
            ''' 
            filters will be stacked as x1,x2,x3,x1,x2,x3,... , therefore, we will mask such that 
            e.g. filter 2 serving x2 can see previous outputs from x3, but not x1
            we will achieve this by building a connections graph, which will zero-out all elements from given channels 
            '''
            # mask A only allows information from lower channels
            Cin = self.mask.shape[1] # input filters
            Cout = self.mask.shape[0] # output filters
            def channel_mask(i_out, i_in): # a map which defines who is allowed to see what
                cout_idx = np.expand_dims(np.arange(Cout) % 2 == i_out, 1)
                cin_idx = np.expand_dims(np.arange(Cin) % 2 == i_in, 0)
                a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
                return a1 * a2

            mask = np.array(self.mask)
            for c in range(2): # mask B allows information from current and lower channels
                mask[channel_mask(c, c), kH // 2, kW // 2] = 0.0 if mask_type == 'A' else 1.0

            mask[channel_mask(0, 1), kH // 2, kW // 2] = 0.0
            self.mask = torch.from_numpy(mask)
    def forward(self, x):
        self.weight.data *= self.mask  # at each forward pass, apply the mask to all the filters (zero-out information about the future)
        return super(MaskedConv2d, self).forward(x)

class MaskedConv2d_h(nn.Conv2d):  # add a mask to the regular Conv2D function, so that it cannot learn about the future
    def __init__(self, mask_type, channels, *args, **kwargs):
        super(MaskedConv2d_h, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}  # mask A is for the first convolutional layer only, all deeper layers use mask B
        _, _, kH, kW = self.weight.size()  # get the size of the convolutional filter
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks

        # spatial masking - prevent information from neighbours
        if mask_type=='A':
            self.mask[:, :, :, -1] = 0  # mask type B allows access to the 'present' pixel, mask A does not

        if channels > 1:
            # channel masking - block information from nearby color channels - ONLY 2 CHANNELS
            ''' 
            filters will be stacked as x1,x2,x3,x1,x2,x3,... , therefore, we will mask such that 
            e.g. filter 2 serving x2 can see previous outputs from x3, but not x1
            we will achieve this by building a connections graph, which will zero-out all elements from given channels 
            '''
            # mask A only allows information from lower channels
            Cin = self.mask.shape[1] # input filters
            Cout = self.mask.shape[0] # output filters
            def channel_mask(i_out, i_in): # a map which defines who is allowed to see what
                cout_idx = np.expand_dims(np.arange(Cout) % 2 == i_out, 1)
                cin_idx = np.expand_dims(np.arange(Cin) % 2 == i_in, 0)
                a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
                return a1 * a2

            mask = np.array(self.mask)
            for c in range(2): # mask B allows information from current and lower channels
                mask[channel_mask(c, c), kH // 2, kW // 2] = 0.0 if mask_type == 'A' else 1.0

            mask[channel_mask(0, 1), kH // 2, kW // 2] = 0.0
            self.mask = torch.from_numpy(mask)
    def forward(self, x):
        self.weight.data *= self.mask  # at each forward pass, apply the mask to all the filters (zero-out information about the future)
        return super(MaskedConv2d_h, self).forward(x)

class DoubleMaskedConv2d(nn.Conv2d):  # adds to regular masked conv2d by masking also the input in subsequent layers (densenet only)
    def __init__(self, mask_type, *args, **kwargs):
        super(DoubleMaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}  # mask A is for the first convolutional layer only, all deeper layers use mask B
        _, _, self.kH, self.kW = self.weight.size()  # get the size of the convolutional filter
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks
        self.mask[:, :, self.kH // 2, self.kW // 2 + (mask_type == 'B'):] = 0  # mask type B allows access to the 'present' pixel, mask A does not
        self.mask[:, :, self.kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask  # at each forward pass, apply the mask to all the filters (zero-out information about the future)
        self.weight.data[0,:, self.kH//2, self.kW//2] *=0 # mask the central pixel of the first filter (which will always be the input in a densent)
        return super(DoubleMaskedConv2d, self).forward(x)

class MaskedPointwiseConv2d(nn.Conv2d):  # adds to regular masked conv2d by masking also the input in subsequent layers (densenet only)
    def __init__(self, *args, **kwargs):
        super(MaskedPointwiseConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data[:,0, 0, 0] *=0 # mask the entirety of the first filter (which will always be the input in a densenet)
        return super(MaskedPointwiseConv2d, self).forward(x)

def gated_activation(input):
    # implement gated activation from Conditional Generation with PixelCNN Encoders
    assert (input.shape[1] % 2) == 0
    a, b = torch.chunk(input, 2, 1) # split input into two equal parts - only works for even number of filters
    a = torch.tanh(a)
    b = torch.sigmoid(b)

    return torch.mul(a,b) # return element-wise (sigmoid-gated) product

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return gated_activation(input)

class Activation(nn.Module):
    def __init__(self, activation_func, *args, **kwargs):
        super().__init__()
        if activation_func == 'gated':
            self.activation = gated_activation
        elif activation_func == 'relu':
            self.activation = F.relu

    def forward(self, input):
        return self.activation(input)

class StackedConvolution(nn.Module):
    def __init__(self, f_in, f_out, padding, dilation, *args, **kwargs):
        super(StackedConvolution, self).__init__(*args, **kwargs)

        #self.v_norm = nn.BatchNorm2d(f_in)
        self.v_Conv2d = nn.Conv2d(f_in, 2 * f_out, (2, 3), 1, padding * (1,1), dilation, bias=True, padding_mode='zeros')
        self.v_to_h_fc = nn.Conv2d(2 * f_out, f_out, 1)
        #self.h_norm = nn.BatchNorm2d(f_in)
        self.h_Conv2d = nn.Conv2d(f_in, f_out, (1, 2), 1, padding * (0,1), dilation, bias=True, padding_mode='zeros')
        self.h_to_h = nn.Conv2d(f_out, f_out, 1)
        self.activation = Activation('gated') # for ReLU, must change number of filters as gated approach halves filters on each application

    def forward(self, v_in, h_in):
        residue = h_in * 1 # residual track

        v_in = self.v_Conv2d(v_in)[:, :, :-1, :]  # remove extra padding
        v_out = self.activation(v_in)
        v_to_h = self.v_to_h_fc(v_in)[:,:,:-1,:] # align v stack to h
        h_in = self.h_Conv2d(h_in)[:, :, :, :-1]  # unpad by 1 on rhs
        h_out = self.activation(torch.cat((h_in, v_to_h), 1))
        h_out = self.h_to_h(h_out) + residue

        return v_out, h_out


class PixelCNN(nn.Module):  # Dense or residual, gated, blocked, dilated PixelCNN with batchnorm
    def __init__(self, filters, initial_convolution_size, dilation, layers, out_maps, padding, channels):
        super(PixelCNN, self).__init__()
        # some vestigial constants from prior implementations (dilation, unpadding...)
        # to run with relu instead of gated activation, have to take out a bunch of factors of 2 (gated activation takes 2*f filters and returns 1*f)

        blocks = 1
        ### initialize constants
        self.padding = padding
        self.layers_per_block = layers
        self.blocks = blocks
        self.layers = int(self.layers_per_block * blocks)
        self.initial_pad = (initial_convolution_size - 1) // 2
        self.main_pad = 1
        initial_filters = filters
        self.input_depth = 1 #for now just 1 channels
        f_in = (np.ones(self.layers + 1) * filters).astype(int)
        f_out = (np.ones(self.layers + 1) * filters).astype(int)
        self.dilation = (np.ones(self.layers) * dilation).astype(int) # not yet in use
        self.unpad = np.zeros(self.layers + 1).astype(int)
        ###
        self.activation = Activation('gated')

        # initial layer
        self.v_initial_convolution = nn.Conv2d(self.input_depth, 2 * initial_filters, (initial_convolution_size//2 + 1, initial_convolution_size), 1, padding * (self.initial_pad + 1, self.initial_pad), padding_mode='zeros', bias=True)
        self.v_to_h_initial = nn.Conv2d(2 * initial_filters, initial_filters,1)
        self.h_initial_convolution = MaskedConv2d_h('A', self.input_depth, self.input_depth, initial_filters, (1, initial_convolution_size//2 + 1), 1, padding * (0, self.initial_pad), padding_mode='zeros', bias=True)
        self.h_to_h_initial = nn.Conv2d(initial_filters, initial_filters, 1)

        # stack layers in blocks
        self.conv_layer = []
        for j in range(blocks):
            self.conv_layer.append([StackedConvolution(f_in[i + j * self.layers_per_block], f_out[i + j * self.layers_per_block], padding, self.dilation[i + j * self.layers_per_block]) for i in range(self.layers_per_block)]) # stacked convolution (no blind spot)
        for j in range(blocks):
            self.conv_layer[j] = nn.ModuleList(self.conv_layer[j])
        self.conv_layer = nn.ModuleList(self.conv_layer)

        #output layers
        self.fc1 = nn.Conv2d(f_out[-1], 256, (1,1)) # final filters are 256, but can really be anything
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
