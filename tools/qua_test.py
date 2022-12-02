'''
Descripttion: 
version: 1.0.0
Author: Gager
Date: 2022-11-23 09:06:01
LastEditors: Gager
'''
import numpy as np


def find_dec_bits_max_min(data, bit_width=8, maximum_bit=32):
    """
    A ragular non-saturated shift-based quantisation mathod. Using max/min values
    :param data:
    :param bit_width:
    :param maximum_bit: maximum decimal bit. Incase sometime bias is too small lead to very large size dec bit
    :return:
    """
    max_val = abs(data.max()) - abs(data.max()/pow(2, bit_width)) # allow very small saturation.
    min_val = abs(data.min()) - abs(data.min()/pow(2, bit_width))
    int_bits = int(np.ceil(np.log2(max(max_val, min_val))))
    dec_bits = (bit_width-1) - int_bits

    return min(dec_bits, maximum_bit)


a = np.random.randn(10, 10)
find_dec_bits_max_min(a)