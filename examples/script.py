import time
import subprocess

examples_path = "/pub/ds3lab-scratch/limhan/bagua/examples/"
datasets = ["cifar_10", "mnist"]
gpu_available = ['2', '4', '5', '7']
gpu_num = [8]
epoch_num = 150
algorithms = ["signSGD", "g_a_r_t"]
lrs = [0.0001, 0.05]


def run_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=None, stderr=None)
    while True:
        if process.poll() == 0:
            print("DONE: ", cmd)
            break
        else:
            time.sleep(1)


def run_experiment(cmd, path):
    print("START: ", cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                               cwd=path)
    while True:
        if process.poll() == 0:
            print("DONE: ", cmd)
            break
        else:
            time.sleep(300)

# def get_cmd():
#     log_name = "logs/{}_{}_gpus_lr_{}_epoch_{}.txt".format(algorithms[1], 2, 0.05, epoch_num)
#     cmd = "python3 -m bagua.distributed.launch --nproc_per_node={} main.py --algorithm {} --lr {} --epochs {} > {} 2>&1".format(
#         2, algorithms[1], 0.05, epoch_num, log_name
#     )
#     print(cmd)

def main():
    for gpu in gpu_num:
        # export_gpu = "export CUDA_VISIBLE_DEVICES={}".format(",".join(gpu_available[:gpu]))
        # run_cmd(export_gpu)
        for dataset in datasets:
            path = examples_path + dataset
            for algo, lr in zip(algorithms, lrs):
                if algo == "signSGD":
                    log_name = "logs_cpp_rt/{}_{}_gpus_lr_{}_epoch_{}.txt".format(algo, gpu, lr, epoch_num)
                    cmd = "python3 -m bagua.distributed.launch --nproc_per_node={} main.py --algorithm {} --lr {} --epochs {} --compress --record_time > {} 2>&1".format(
                        gpu, algo, lr, epoch_num, log_name
                    )
                    run_experiment(cmd, path)

                    log_name = "logs_cpp_rt/{}_{}_gpus_lr_{}_epoch_{}_without_compression.txt".format(algo, gpu, lr, epoch_num)
                    cmd = "python3 -m bagua.distributed.launch --nproc_per_node={} main.py --algorithm {} --lr {} --epochs {} --record_time > {} 2>&1".format(
                        gpu, algo, lr, epoch_num, log_name
                    )
                    run_experiment(cmd, path)

                else:
                    log_name = "logs_allreduce_rt/{}_{}_gpus_lr_{}_epoch_{}.txt".format(algo, gpu, lr, epoch_num)
                    cmd = "python3 -m bagua.distributed.launch --nproc_per_node={} main.py --algorithm {} --lr {} --epochs {} --record_time > {} 2>&1".format(
                        gpu, algo, lr, epoch_num, log_name
                    )
                    run_experiment(cmd, path)


main()
