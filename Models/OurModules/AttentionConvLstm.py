# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:26:28 2019
@author: Xingguang Zhang
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class GateConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True, init_method = 'xavier_normal_'):
        super(GateConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4
        self.init_method = init_method
        self.padding = int((kernel_size - 1) / 2)
            
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc_d = nn.Conv2d(self.input_channels, self.input_channels, self.kernel_size, \
                               1, self.padding, bias=True, groups = self.input_channels)
        self.Wxc_p = nn.Conv2d(self.input_channels, self.hidden_channels, (1,1), 1, 0, bias=False)
        self.Whc_d = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,\
                               1, self.padding, bias=False, groups = self.hidden_channels)
        self.Whc_p = nn.Conv2d(self.hidden_channels, self.hidden_channels, (1,1), 1, 0, bias=False)        
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.xgpooling = nn.AdaptiveAvgPool2d(1)
        self.hgpooling = nn.AdaptiveAvgPool2d(1)

        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                getattr(nn.init, self.init_method)(w.weight)

    def forward(self, x, h, c):
        x_global = self.xgpooling(x)
        h_global = self.hgpooling(h)
        ci = torch.sigmoid(self.Wxi(x_global) + self.Whi(h_global))
        cf = torch.sigmoid(self.Wxf(x_global) + self.Whf(h_global))
        co = torch.sigmoid(self.Wxo(x_global) + self.Who(h_global))
        G = torch.tanh(self.Wxc_p(self.Wxc_d(x)) + self.Whc_p(self.Whc_d(h)))
        cc = cf * c + ci * G
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                    Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())
        else:
            return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                    Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))

class AConvLSTMCell_b(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True, init_method = 'xavier_normal_'):
        super(AConvLSTMCell_b, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4
        self.init_method = init_method
        self.padding = int((kernel_size - 1) / 2)
        
        self.Wxa_d = nn.Conv2d(self.input_channels, self.input_channels, self.kernel_size, \
                               1, self.padding, bias=True, groups = self.input_channels)
        self.Wxa_p = nn.Conv2d(self.input_channels, self.hidden_channels, (1,1), 1, 0, bias=False)
        self.Wha_d = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, \
                               1, self.padding, bias=True, groups = self.input_channels)
        self.Wha_p = nn.Conv2d(self.hidden_channels, self.hidden_channels, (1,1), 1, 0, bias=False)
        self.Wz = nn.Conv2d(self.hidden_channels, self.input_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc_d = nn.Conv2d(self.input_channels, self.input_channels, self.kernel_size, \
                               1, self.padding, bias=True, groups = self.input_channels)
        self.Wxc_p = nn.Conv2d(self.input_channels, self.hidden_channels, (1,1), 1, 0, bias=True)
        self.Whc_d = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,\
                               1, self.padding, bias=False, groups = self.hidden_channels)
        self.Whc_p = nn.Conv2d(self.hidden_channels, self.hidden_channels, (1,1), 1, 0, bias=False)        
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=False)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.xgpooling = nn.AdaptiveAvgPool2d(1)
        self.hgpooling = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)

        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                getattr(nn.init, self.init_method)(w.weight)

    def SoftmaxPixel(self, s):
        batch,channel,height,weight = s.size()
        return self.softmax(s.view(batch,channel,-1)).view(batch,channel,height,weight)

    def forward(self, x, h, c):
        Zt = self.Wz(torch.tanh(self.Wxa_p(self.Wxa_d(x)) + self.Wha_p(self.Wha_d(h))))
        At = self.SoftmaxPixel(Zt)
        x = At * x
        x_global = self.xgpooling(x)
        h_global = self.hgpooling(h)
        ci = torch.sigmoid(self.Wxi(x_global) + self.Whi(h_global))
        cf = torch.sigmoid(self.Wxf(x_global) + self.Whf(h_global))
        co = torch.sigmoid(self.Wxo(x_global) + self.Who(h_global))
        G = torch.tanh(self.Wxc_p(self.Wxc_d(x)) + self.Whc_p(self.Whc_d(h)))     
        cc = cf * c + ci * G
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                    Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())
        else:
            return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                    Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))

class AConvLSTMCell_c(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True, init_method = 'xavier_normal_'):
        super(AConvLSTMCell_c, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4
        self.init_method = init_method
        self.padding = int((kernel_size - 1) / 2)
        
        self.Wxa_d = nn.Conv2d(self.input_channels, self.input_channels, self.kernel_size, \
                               1, self.padding, bias=True, groups = self.input_channels)
        self.Wxa_p = nn.Conv2d(self.input_channels, self.hidden_channels, (1,1), 1, 0, bias=False)
        self.Wha_d = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, \
                               1, self.padding, bias=True, groups = self.input_channels)
        self.Wha_p = nn.Conv2d(self.hidden_channels, self.hidden_channels, (1,1), 1, 0, bias=False)
        self.Wz = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc_d = nn.Conv2d(self.input_channels, self.input_channels, self.kernel_size, \
                               1, self.padding, bias=True, groups = self.input_channels)
        self.Wxc_p = nn.Conv2d(self.input_channels, self.hidden_channels, (1,1), 1, 0, bias=True)
        self.Whc_d = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,\
                               1, self.padding, bias=False, groups = self.hidden_channels)
        self.Whc_p = nn.Conv2d(self.hidden_channels, self.hidden_channels, (1,1), 1, 0, bias=True)        
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.softmax= nn.Softmax(dim = 2)
        self.xgpooling = nn.AdaptiveAvgPool2d(1)
        self.hgpooling = nn.AdaptiveAvgPool2d(1)
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                getattr(nn.init, self.init_method)(w.weight)
                
    def SoftmaxPixel_Max(self, s):
        batch,channel,height,weight = s.size()
        newS = self.softmax(s.view(batch,channel,-1))
        MaxS,_ = torch.max(newS, dim = 2, keepdim = True,out=None)
        newS = newS / MaxS
        return newS.view(batch,channel,height,weight)

    def forward(self, x, h, c):
        Zt = self.Wz(torch.tanh(self.Wxa_p(self.Wxa_d(x)) + self.Wha_p(self.Wha_d(h))))
        ci = self.SoftmaxPixel_Max(Zt)                
        x_global = self.xgpooling(x)
        h_global = self.hgpooling(h)
        cf = torch.sigmoid(self.Wxf(x_global) + self.Whf(h_global))
        co = torch.sigmoid(self.Wxo(x_global) + self.Who(h_global))
        G = torch.tanh(self.Wxc_p(self.Wxc_d(x)) + self.Whc_p(self.Whc_d(h)))
        cc = cf * c + ci * G
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                    Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())
        else:
            return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                    Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))

class AConvLSTMCell_d(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True, init_method = 'xavier_normal_'):
        super(AConvLSTMCell_d, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4
        self.init_method = init_method
        self.padding = int((kernel_size - 1) / 2)
        
        self.Wxa_d = nn.Conv2d(self.input_channels, self.input_channels, self.kernel_size, \
                               1, self.padding, bias=True, groups = self.input_channels)
        self.Wxa_p = nn.Conv2d(self.input_channels, self.hidden_channels, (1,1), 1, 0, bias=False)
        self.Wha_d = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, \
                               1, self.padding, bias=True, groups = self.input_channels)
        self.Wha_p = nn.Conv2d(self.hidden_channels, self.hidden_channels, (1,1), 1, 0, bias=False)
        self.Wz = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc_d = nn.Conv2d(self.input_channels, self.input_channels, self.kernel_size, \
                               1, self.padding, bias=True, groups = self.input_channels)
        self.Wxc_p = nn.Conv2d(self.input_channels, self.hidden_channels, (1,1), 1, 0, bias=True)
        self.Whc_d = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,\
                               1, self.padding, bias=False, groups = self.hidden_channels)
        self.Whc_p = nn.Conv2d(self.hidden_channels, self.hidden_channels, (1,1), 1, 0, bias=True)        
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.softmax= nn.Softmax(dim = 2)
        self.xgpooling = nn.AdaptiveAvgPool2d(1)
        self.hgpooling = nn.AdaptiveAvgPool2d(1)

        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                getattr(nn.init, self.init_method)(w.weight)

    def SoftmaxPixel_Max(self, s):
        batch,channel,height,weight = s.size()
        newS = self.softmax(s.view(batch,channel,-1))
        MaxS,_ = torch.max(newS, dim = 2, keepdim = True,out=None)
        newS = newS / MaxS
        return newS.view(batch,channel,height,weight)
    
    def forward(self, x, h, c):
        Zt = self.Wz(torch.tanh(self.Wxa_p(self.Wxa_d(x)) + self.Wha_p(self.Wha_d(h))))
        co = self.SoftmaxPixel_Max(Zt)
        x_global = self.xgpooling(x)
        h_global = self.hgpooling(h)
        ci = torch.sigmoid(self.Wxi(x_global) + self.Whi(h_global))
        cf = torch.sigmoid(self.Wxf(x_global) + self.Whf(h_global))
#        co = torch.sigmoid(self.Wxo(x_global) + self.Who(h_global))
        G = torch.tanh(self.Wxc_p(self.Wxc_d(x)) + self.Whc_p(self.Whc_d(h)))
        cc = cf * c + ci * G
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                    Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())
        else:
            return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                    Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))
        
class AttenConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, \
                 bias=True, init_method = 'xavier_normal_', AttenMethod = 'a', expand_x= 28, expand_y=28):
        super(AttenConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.init_method = init_method
        self.expand_x = expand_x
        self.expand_y = expand_y
        #self.effective_step = effective_step

        if AttenMethod == 'a':
            cell0 = GateConvLSTMCell(self.input_channels[0], self.hidden_channels[0], \
                                    self.kernel_size, self.bias, self.init_method)
            setattr(self, 'cell0', cell0)
            
        elif AttenMethod == 'b':
            cell0 = AConvLSTMCell_b(self.input_channels[0], self.hidden_channels[0], \
                                    self.kernel_size, self.bias, self.init_method)
            setattr(self, 'cell0', cell0)
            
        elif AttenMethod == 'c':
            cell0 = AConvLSTMCell_c(self.input_channels[0], self.hidden_channels[0], \
                                    self.kernel_size, self.bias, self.init_method)
            setattr(self, 'cell0', cell0)
            
        elif AttenMethod == 'd':
            cell0 = AConvLSTMCell_d(self.input_channels[0], self.hidden_channels[0], \
                                    self.kernel_size, self.bias, self.init_method)
            setattr(self, 'cell0', cell0)
            
        cell1 = GateConvLSTMCell(self.input_channels[1], self.hidden_channels[1], \
                                 self.kernel_size, self.bias, self.init_method)
        setattr(self, 'cell1', cell1)
        
    def forward(self, input):
        internal_state = []
        if input.is_cuda:
            outputs = torch.randn(input.shape[0], self.hidden_channels[-1], self.step, self.expand_x, self.expand_y).cuda()
        else:
            outputs = torch.randn(input.shape[0], self.hidden_channels[-1], self.step, self.expand_x, self.expand_y)
        for step in range(self.step):
            x = input[:,:,step,:,:]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize,\
                    hidden=self.hidden_channels[i], shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            outputs[:,:,step,:,:] = x
        return outputs, (x, new_c)