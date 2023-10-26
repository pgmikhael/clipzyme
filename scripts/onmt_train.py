import argparse
import subprocess
import multiprocessing
import os
from rich import print

parser = argparse.ArgumentParser(description="Dispatcher.")
parser.add_argument(
    "--data_folder_names",
    type=str,
    required=True,
    nargs="+",
    default=[],
    help="name of dataset folders",
)
parser.add_argument(
    "--available_gpus",
    type=str,
    required=True,
    nargs="+",
    default=[],
    help="gpus to use for train",
)


rbt_dir = "/Mounts/rbg-storage1/users/pgmikhael/rbt-preprocess"


def onmt_process_data(experiment_name):
    return f"""onmt_preprocess \
        -train_src "{rbt_dir}/datasets/uspto_dataset/src-train.txt" "{rbt_dir}/datasets/{experiment_name}/src-train.txt" \
        -train_tgt "{rbt_dir}/datasets/uspto_dataset/tgt-train.txt" "{rbt_dir}/datasets/{experiment_name}/tgt-train.txt" \
        -train_ids uspto transfer \
        -valid_src "{rbt_dir}/datasets/{experiment_name}/src-valid.txt" \
        -valid_tgt "{rbt_dir}/datasets/{experiment_name}/tgt-valid.txt" \
        -save_data "{rbt_dir}/datasets/{experiment_name}/onmt-preprocess" \
        -src_seq_length 3000 -tgt_seq_length 3000 \
        -src_vocab_size 3000 -tgt_vocab_size 3000 \
        -share_vocab"""


def train(experiment_name):
    return f"""onmt_train \
        -data {rbt_dir}/datasets/{experiment_name}/onmt-preprocess  \
        -save_model {rbt_dir}/datasets/models/{experiment_name}\
        -data_ids uspto transfer\
        --data_weights 9 1\
        -seed 42\
        -gpu_ranks 0\
        -world_size 1\
        -train_steps 250000\
        -save_checkpoint_steps 50000 \
        -param_init 0\
        -param_init_glorot\
        -max_generator_batches 32\
        -batch_size 32768\
        -batch_type tokens\
        -normalization tokens\
        -max_grad_norm 0\
        -accum_count 1\
        -optim adam\
        -adam_beta1 0.9\
        -adam_beta2 0.998\
        -decay_method noam\
        -warmup_steps 8000\
        -learning_rate 2\
        -label_smoothing 0.1\
        -layers 6\
        -rnn_size 512\
        -word_vec_size 512\
        -encoder_type transformer\
        -decoder_type transformer\
        -dropout 0.1\
        -position_encoding\
        -share_embeddings\
        -global_attention general\
        -global_attention_function softmax\
        -self_attn_type scaled-dot\
        -heads 8\
        -transformer_ff 2048\
        --tensorboard\
        --tensorboard_log_dir {rbt_dir}/datasets/models/{experiment_name}
        """


def predict(experiment_name):
    return f"""onmt_translate \
        -model {rbt_dir}/datasets/models/{experiment_name}_step_250000.pt \
        -src "{rbt_dir}/datasets/{experiment_name}/src-test.txt" \
        -output "{rbt_dir}/datasets/{experiment_name}/tgt-pred.txt" \
        -n_best 10 \
        -beam_size 10 \
        -max_length 300 \
        -batch_size 64 \
        -gpu 0
        """


def launch_experiment(gpu, datafolder):
    """
    Launch an experiment and direct logs and results to a unique filepath.

    Args:
        script (str): file name to run as main
        gpu (str): gpu this worker can access.
        flag_string (str): arguments and values as a single blob.

    Returns:
        results_path (str): path to saved args pickle file
        log_path (str): path to logs
    """
    if not os.path.isdir(f"{rbt_dir}/datasets/models/{datafolder}"):
        os.makedirs(f"{rbt_dir}/datasets/models/{datafolder}")

    log_path = f"{rbt_dir}/datasets/models/{datafolder}/log.txt"

    print("Launched experiment for: {}".format(datafolder))

    # run onmt_process
    print()
    print(onmt_process_data(datafolder))
    subprocess.call(onmt_process_data(datafolder), shell=True)

    # run train
    train_str = "CUDA_VISIBLE_DEVICES={} {} > {} 2>&1".format(
        gpu, train(datafolder), log_path
    )
    print()
    print(train_str)
    subprocess.call(train_str, shell=True)

    # run predict
    test_str = "CUDA_VISIBLE_DEVICES={} {}".format(gpu, predict(datafolder))
    print()
    print(test_str)
    subprocess.call(test_str, shell=True)

    return log_path


def worker(gpu, job_queue, done_queue):
    """
    Worker thread for each gpu. Consumes all jobs and pushes results to done_queue.

    Args:
        script (str): file name to run as main
        gpu (str): gpu this worker can access.
        job_queue (Queue): queue of available jobs.
        done_queue (Queue): queue where to push results.
    """

    while not job_queue.empty():
        foldername = job_queue.get()
        if foldername is None:
            return
        done_queue.put(launch_experiment(gpu, foldername))


if __name__ == "__main__":
    args = parser.parse_args()

    assert len(args.data_folder_names) == len(args.available_gpus)

    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for job in args.data_folder_names:
        job_queue.put(job)
    print("Launching Dispatcher with {} jobs!".format(len(args.data_folder_names)))
    print()

    for gpu in args.available_gpus:
        print("Start gpu worker {}".format(gpu))
        multiprocessing.Process(
            target=worker,
            args=(gpu, job_queue, done_queue),
        ).start()
        print()

    for i in range(len(args.data_folder_names)):
        log_path = done_queue.get()  # .txt (stderr/out) files
        print("Logs are located at: {}".format(log_path))
