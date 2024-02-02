import dl_assignment_7_common as common

from multiprocessing import Process

if __name__ == '__main__':
    trials = 5
    p_list = []

    # for pr_rate in [1, 0.644, 0.417, 0.271, 0.178, 0.118, 0.08, 0.055]:
    for pr_rate in [1, 0.417, 0.178, 0.08]:
        p = Process(target=common.iterative_global_pruning,
                    args=('resnet18', 'cifar10', 500, 5, 1 - pr_rate, trials, False))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()

    # for pr_rate in [1, 0.644, 0.417, 0.271, 0.178, 0.118, 0.08, 0.055]:
    for pr_rate in [1, 0.417, 0.178, 0.08]:
        p = Process(target=common.iterative_global_pruning,
                    args=('resnet18', 'cifar10', 500, 5, 1 - pr_rate, trials, True))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()
