import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(num_filters * H / 2.0 * W / 2.0, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.rand(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, ar_cache = affine_relu_forward(out, W2, b2)
        scores, a_cache = affine_forward(out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)

        dx, grads['W3'], grads['b3'] = affine_backward(dscores, a_cache)
        dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, ar_cache)
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, crp_cache)
        for i in xrange(1, 4):
            W = self.params['W' + str(i)]
            loss += 0.5 * self.reg * np.sum(W * W)
            grads['W' + str(i)] += self.reg * W
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class ConvNet(object):
    def __init__(self, num_filters_each_layer, filter_size_each_layer,
                 hidden_dim_each_layer, use_max_pool=False, use_sbatchnorm=False, use_batchnorm=False,
                 input_dim=(3, 32, 32), num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):

        # The architecture of this network is (conv-(sbatchnorm: optional)-relu) xN-(maxpool: optional), affine-(batchnorm: optional)-relu x M, affine-softmax
        # For conv. layers, input to each layer is zero padded, kernel moves with stride = 1
        # Also, max. pooling has stride 2 and size of kernel = 2

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_max_pool = use_max_pool
        self.use_batchnorm = use_batchnorm
        self.use_sbatchnorm = use_sbatchnorm
        self.filter_size_each_layer = filter_size_each_layer
        self.num_filters_each_layer = num_filters_each_layer
        hidden_dim_each_layer.append(num_classes)
        self.hidden_dim_each_layer = hidden_dim_each_layer

        C, H, W = input_dim
        num_conv = len(num_filters_each_layer)
        num_affine = len(hidden_dim_each_layer)
        wbcount = 0
        sgbcount = 0
        gbcount = 0

        for i in xrange(num_conv):
            wbcount +=1
            new_w = 'W' + str(wbcount)
            new_b = 'b' + str(wbcount)
            if i == 0:
                self.params[new_w] = weight_scale * np.random.randn(num_filters_each_layer[i], C,
                                                                               filter_size_each_layer[i],
                                                                               filter_size_each_layer[i])
            else:
                self.params[new_w] = weight_scale * np.random.randn(num_filters_each_layer[i],
                                                                               num_filters_each_layer[i - 1],
                                                                               filter_size_each_layer[i],
                                                                               filter_size_each_layer[i])
            self.params[new_b] = np.zeros(num_filters_each_layer[i])
            if use_sbatchnorm:
                sgbcount += 1
                strsgbcount = str(sgbcount)
                self.params['s_gamma' + strsgbcount] = np.ones(num_filters_each_layer[i])
                self.params['s_beta' + strsgbcount] = np.zeros(num_filters_each_layer[i])


        ######## Affine layer from conv layers.
        if self.use_max_pool:
            factor = 1
        else:
            factor = 0
        wbcount+=1
        strwbcount = str(wbcount)
        self.params['W' + strwbcount] = weight_scale * np.random.randn(
            num_filters_each_layer[num_conv - 1] * H / (2.0 ** factor) * W / (2.0 ** factor), hidden_dim_each_layer[0])
        self.params['b' + strwbcount] = np.zeros(hidden_dim_each_layer[0])
        if use_batchnorm:
            gbcount+=1
            strgbcount = str(gbcount)
            self.params['gamma' + strgbcount] = np.ones(hidden_dim_each_layer[0])
            self.params['beta' + strgbcount] = np.zeros(hidden_dim_each_layer[0])


        ####### Remaining affine layers.
        for j in xrange(num_affine-1):
            wbcount += 1
            strwbcount = str(wbcount)
            new_w = 'W' + strwbcount
            new_b = 'b' + strwbcount
            self.params[new_w] = weight_scale * np.random.rand(hidden_dim_each_layer[j],
                                                                                     hidden_dim_each_layer[j + 1])
            self.params[new_b] = np.zeros(hidden_dim_each_layer[j + 1])
            if j != num_affine-2 and use_batchnorm:
                gbcount += 1
                strgbcount = str(gbcount)
                self.params['gamma' + strgbcount] = np.ones(hidden_dim_each_layer[j + 1])
                self.params['beta' + strgbcount] = np.zeros(hidden_dim_each_layer[j + 1])

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(num_affine - 1)]

        self.sbn_params = []
        if self.use_sbatchnorm:
            self.sbn_params = [{'mode': 'train'} for i in xrange(num_conv)]

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        # Evaluate loss and gradient for the convolutional network.

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # Forward Pass
        num_conv = len(self.num_filters_each_layer)
        num_affine = len(self.hidden_dim_each_layer) + 1

        input_to_layer = X
        conv_caches = []
        affine_caches = []
        for i in xrange(num_conv):
            conv_param = {'stride': 1, 'pad': (self.filter_size_each_layer[i] - 1) / 2}
            if i == num_conv - 1 and self.use_max_pool:
                if self.use_sbatchnorm:
                    input_to_layer, conv_cache = conv_bnorm_relu_pool_forward(input_to_layer,
                                                                              self.params['W' + str(i + 1)],
                                                                              self.params['b' + str(i + 1)], conv_param,
                                                                              pool_param,
                                                                              self.params['s_gamma' + str(i + 1)],
                                                                              self.params['s_beta' + str(i + 1)],
                                                                              self.sbn_params[i])
                else:
                    input_to_layer, conv_cache = conv_relu_pool_forward(input_to_layer, self.params['W' + str(i + 1)],
                                                                        self.params['b' + str(i + 1)], conv_param,
                                                                        pool_param,
                                                                        self.params['s_gamma' + str(i + 1)],
                                                                        self.params['s_beta' + str(i + 1)])

            elif self.use_sbatchnorm:
                input_to_layer, conv_cache = conv_bnorm_relu_forward(input_to_layer, self.params['W' + str(i + 1)],
                                                                     self.params['b' + str(i + 1)], conv_param,
                                                                     self.params['s_gamma' + str(i + 1)],
                                                                     self.params['s_beta' + str(i + 1)],
                                                                     self.sbn_params[i])
            else:
                input_to_layer, conv_cache = conv_relu_forward(input_to_layer, self.params['W' + str(i + 1)],
                                                               self.params['b' + str(i + 1)], conv_param)
            conv_caches.append(conv_cache)

        scores = None
        for j in xrange(num_affine-1):
            if j == num_affine - 2:
                scores, affine_cache = affine_forward(input_to_layer, self.params['W' + str(j + num_conv+1)],
                                                      self.params['b' + str(j + num_conv+1)])
            elif self.use_batchnorm:
                input_to_layer, affine_cache = affine_batchnorm_relu_forward(input_to_layer,
                                                                             self.params['W' + str(j + num_conv+1)],
                                                                             self.params['b' + str(j + num_conv+1)],
                                                                             self.params['gamma' + str(j + 1)],
                                                                             self.params['beta' + str(j + 1)],
                                                                             self.bn_params[j])
            else:
                input_to_layer, affine_cache = affine_relu_forward(input_to_layer, self.params['W' + str(j + num_conv+1)],
                                                                   self.params['b' + str(j + num_conv+1)])
            affine_caches.append(affine_cache)

        if y is None:
            return scores

        # Loss and backprop.
        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)

        dx = None
        for i in xrange(num_affine - 2, -1, -1):
            if i == num_affine - 2:
                dx, grads['W' + str(i+num_conv+1)], grads['b' + str(i+num_conv+1)] = affine_backward(dscores,affine_caches[i])
            elif self.use_batchnorm:
                dx, grads['W' + str(i+num_conv+1)], grads['b' + str(i+num_conv+1)], grads['gamma' + str(i+1)], \
                grads['beta' + str(i+1)] = affine_batchnorm_relu_backward(dx, affine_caches[i])
            else:
                dx, grads['W' + str(i+num_conv+1)], grads['b' + str(i+num_conv+1)] = affine_relu_backward(dx, affine_caches[i])

        for i in xrange(num_conv-1, -1, -1):
            if i==num_conv-1 and self.use_max_pool:
                if self.use_sbatchnorm:
                    dx, grads['W' + str(i + 1)], grads['b' + str(i + 1)], grads['s_gamma' + str(i + 1)], \
                    grads['s_beta' + str(i + 1)] = conv_bnorm_relu_pool_backward(dx, conv_caches[i])
                else:
                    dx, grads['W' + str(i + 1)], grads['b' + str(i + 1)] = conv_relu_pool_backward(dx, conv_caches[i])
            elif self.use_sbatchnorm:
                dx, grads['W' + str(i+1)], grads['b' + str(i+1)], grads['s_gamma' + str(i+1)], \
                grads['s_beta' + str(i+1)] = conv_bnorm_relu_backward(dx, conv_caches[i])
            else:
                dx, grads['W' + str(i+1)], grads['b' + str(i+1)] = conv_relu_backward(dx, conv_caches[i])

        for i in xrange(1, num_conv + num_affine):
            W = self.params['W' + str(i)]
            loss += 0.5 * self.reg * np.sum(W * W)
            grads['W' + str(i)] += self.reg * W
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
