import dl_assignment_7_common as common

from multiprocessing import Process

if __name__ == '__main__':
    trials = 5
    p_list = []

    for pr_rate in [1, 0.513, 0.211, 0.07, 0.036, 0.019]:
        p = Process(target=common.iterative_pruning, args=('lenet', 'mnist', 500, 10, 1 - pr_rate, trials))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()

