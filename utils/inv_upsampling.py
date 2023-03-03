#!/usr/bin/env python

# from https://github.com/jleuschn/cinn_for_imaging

from utils.iunets.layers import InvertibleDownsampling2D, InvertibleUpsampling2D
import FrEIA.modules as Fm
from torch.nn.modules.utils import _pair

class InvertibleDownsampling(Fm.InvertibleModule):
    def __init__(self, dims_in, stride=2, method='cayley', init='haar',
                 learnable=True, *args, **kwargs):
        super().__init__(dims_in)
        self.stride = tuple(_pair(stride))
        self.invertible_downsampling = InvertibleDownsampling2D(in_channels=dims_in[0][0],
                                                                stride=stride,
                                                                method=method,
                                                                init=init,
                                                                learnable=learnable,
                                                                *args,
                                                                **kwargs)

    def forward(self, x, rev=False, jac=True):
        x = x[0]
        log_jac_det = 0.
        if not rev:
            x = self.invertible_downsampling.forward(x)
        else:
            x = self.invertible_downsampling.inverse(x)

        return (x,), log_jac_det

    
    def output_dims(self, input_dims):
        """
        Calculates the output dimension of the invertible downsampling.
        Currently, only a stride of 2 is supported
        Parameters
        ----------
        input_dims : TYPE
            DESCRIPTION.
        Returns
        -------
        list
            DESCRIPTION.
        """
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c*self.stride[0] * self.stride[1], w//self.stride[0], h//self.stride[1]
        self.elements = c*w*h
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]


class InvertibleUpsampling(Fm.InvertibleModule):
    def __init__(self, dims_in, stride=2, method='cayley', init='haar',
                 learnable=True, *args, **kwargs):
        super().__init__(dims_in)
        self.stride = tuple(_pair(stride))
        self.invertible_upsampling = InvertibleUpsampling2D(in_channels=dims_in[0][0],
                                                                stride=stride,
                                                                method=method,
                                                                init=init,
                                                                learnable=learnable,
                                                                *args,
                                                                **kwargs)


    def forward(self, x, rev=False, jac=True):
        x = x[0]
        log_jac_det = 0.
        if not rev:
            x = self.invertible_upsampling.forward(x)
        else:
            x = self.invertible_upsampling.inverse(x)

        return (x,), log_jac_det

    
    def output_dims(self, input_dims):
        """
        Calculates the output dimension of the invertible downsampling.
        Currently, only a stride of 2 is supported
        Parameters
        ----------
        input_dims : TYPE
            DESCRIPTION.
        Returns
        -------
        list
            DESCRIPTION.
        """
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c//(self.stride[0] * self.stride[1]), w*self.stride[0], h*self.stride[1]
        self.elements = c*w*h
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]

