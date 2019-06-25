import os
import torch as tr
from surprise_based.net_CVAE import *

def save_checkpoint(state, pathname='checkpoint.pth.tar'):
    tr.save(state, pathname)

device = 'cpu'
path = os.path.abspath('VAE_zsa/')
net = CVAE(name='CVAE_Model').to(device)
safeBuf = Memory(MemLen=1e5)

for i in range(20):
    actions = data[:, 0]
    if len(actions.shape) == 1:
        actions = actions[:, np.newaxis]
    loss, MSE, KLD = net.train_step(actions, data[:, 1:21], data[:, 21:], device)
    if i % 20 == 0 and i != 0:
        print("step: ", i, ";    loss: ", loss.data,
              ';    MSE loss: ', MSE.data, ';    KL_D: ', KLD.data)

    if i % 20 == 0 and i != 0:
        filename_ckpt = net.name + '_ckpt_' + str(i) + '.pth.tar'
        pathname_ckpt = os.path.join(path, filename_ckpt)
        save_checkpoint({
            'state_dict': net.state_dict(),
            'optimizer': net.opt.state_dict(),
            'ckpt_episode': i
        }, pathname_ckpt)

filename_trained = net.name + '.pth.tar'
pathname_trained = os.path.join(path, filename_trained)

tr.save(net.state_dict(), pathname_trained)