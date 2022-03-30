# -*- coding: utf-8 -*-

import numpy as np

for ilayer in range(20):
    ilay, jlay, Tlay, scale_lay, indent, word1, word2, word3, word4, \
        word5,word6 = np.loadtxt('test_read_drv.txt',skiprows=ilayer,
                        max_rows=1,unpack=True,dtype=str)
    print(ilay, jlay, Tlay, scale_lay, indent, word1, word2, word3, word4,word5,word6)