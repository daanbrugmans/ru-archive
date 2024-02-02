import torch

from dl_assignment_7_common import *

if __name__ == '__main__':
    net = create_network('lenet')
    prune(net, 0.5)
    prune(net, 0.5)
    net.load_state_dict(torch.load("checkpoints/model-experiment1-lenet-mnist-0.589-0.000-one-shot-1-best.pth"))
    print()