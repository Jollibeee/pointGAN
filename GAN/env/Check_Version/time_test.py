# -*- coding: utf-8 -*-
import time

time = 9
epoch = 1000

mm, ss = divmod(time,60)
hh, mm = divmod(mm, 60)
print('%2d:%2d:%2d / 1 epoch' %(hh, mm, ss))


total = time * epoch
m, s = divmod(total, 60)
h, m = divmod(m, 60)

print('%2d:%2d:%2d / %d epoch' %(h, m, s, epoch))



''' pointnet
AE 500
    ====== chair ===== 
    0: 0:21 / 1 epoch
    2:55: 0 / 500 epoch 예상
    2:54:44 / 500 epoch 실

'''
