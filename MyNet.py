import math

import numpy as np
import paddle


class ConvPool(paddle.nn.Layer):
    """卷积+池化"""

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 groups,
                 conv_stride=1,
                 conv_padding=1,
                 ):
        super(ConvPool, self).__init__()

        for i in range(groups):
            self.add_sublayer(  # 添加子层实例
                'bb_%d' % i,
                paddle.nn.Conv2D(  # layer
                    in_channels=num_channels,  # 通道数
                    out_channels=num_filters,  # 卷积核个数
                    kernel_size=filter_size,  # 卷积核大小
                    stride=conv_stride,  # 步长
                    padding=conv_padding,  # padding
                )
            )
            self.add_sublayer(
                'relu%d' % i,
                paddle.nn.ReLU()
            )
            num_channels = num_filters

        self.add_sublayer(
            'Maxpool',
            paddle.nn.MaxPool2D(
                kernel_size=pool_size,  # 池化核大小
                stride=pool_stride  # 池化步长
            )
        )

    def forward(self, inputs):
        x = inputs
        for prefix, sub_layer in self.named_children():
            # print(prefix,sub_layer)
            x = sub_layer(x)
        return x


class MyNet(paddle.nn.Layer):

    def __init__(self, length):
        super(MyNet, self).__init__()
        self.length = length
        
        self.firstfc01 = paddle.nn.Linear(120,120,bias_attr=True)
        self.firstfc02 = paddle.nn.Linear(120,256,bias_attr=True)

        self.attention1 = paddle.nn.Linear(512, 2048, bias_attr=True)
        self.attention2 = paddle.nn.Linear(512, 2048, bias_attr=True)
        self.attention3 = paddle.nn.Linear(512, 2048, bias_attr=True)

        
        #32*32
        self.convpool11 = ConvPool(2*self.length+1,64,3,2,2,2)
        #16*16
        self.convpool12 = ConvPool(64,128,3,2,2,2)
        #8*8
        self.convpool13 = ConvPool(128,256,3,2,2,3)
        #4*4
        self.convpool14 = ConvPool(256,512,3,2,2,3)
        #512*2*2
       
        self.pool_5_shape = 512*2*2
        self.fc01 = paddle.nn.Linear(self.pool_5_shape, 1024, bias_attr=True)
        self.fc02 = paddle.nn.Linear(1024, 1024, bias_attr=True)
        self.fc03 = paddle.nn.Linear(1024, 2, bias_attr=True)

    def forward(self, inputs, label=None):
        """前向计算"""
        m = paddle.nn.Dropout(p=0.2)
        inputs0 = paddle.cast(inputs[0], dtype='float32')
	
        inputsgeo = paddle.reshape(inputs0, shape=[-1, 2*self.length+1, 6*20])
        relu = paddle.nn.ReLU()
        out = self.firstfc01(inputsgeo)
        out = self.firstfc02(out)
        #out = m(out)
        #11*256
        position_encoding = paddle.cast(inputs[1], dtype='float32')

        input3 = paddle.concat(x=[out, paddle.cast(position_encoding, dtype='float32')], axis=-1)

        query = self.attention1(input3)
        key = self.attention2(input3)
        value = self.attention3(input3)

        key = paddle.reshape(key, [-1, 2048, 2*self.length+1])
        out = paddle.matmul(query, key)
        out = paddle.scale(out, 1 / math.sqrt(2048))
        out = paddle.nn.functional.softmax(out)
        out = paddle.matmul(out, value)
        maxpool01 = paddle.nn.layer.MaxPool1D(kernel_size=2, stride=2, padding=0)
        out = maxpool01(out)
        #out = m(out)
        out = relu(out)
        out = paddle.reshape(out, shape=[-1, 2*self.length+1, 32, 32])

        out = self.convpool11(out)
        out = self.convpool12(out)
        out = self.convpool13(out)
        out = self.convpool14(out)
        #out = m(out)
        out = relu(out)

        out = paddle.reshape(out, shape = [-1, 512*2*2])
        
        out = self.fc01(out)
        out = self.fc02(out)
        out = self.fc03(out)
        out = paddle.nn.functional.softmax(out)
        if label is not None:
            acc = paddle.metric.accuracy(input=out, label=label)
            return out, acc
        else:
            return out
