import dl_assignment_7_common as common

from torch.optim.adam import Adam
from multiprocessing import Process

if __name__ == '__main__':
    p_list = []

    for pr_rate in [1, 0.411, 0.169, 0.070, 0.029, 0.012, 0.005, 0.002]:
        p = Process(target=common.experiment_section_1, args=('lenet', 'mnist', Adam, 0.0012, 1 - pr_rate, 0, 5))
        p.start()
        p_list.append(p)
    
    for p in p_list:
        p.join()

