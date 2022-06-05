import json
import os
import numpy as np
import argparse


def sort_by_time(dirs):
    mtd_times = []
    for direc in dirs:
        path = os.path.join(direc, 'job_info.json')
        if not os.path.isfile(path):
            continue
        with open(path, 'r') as f:
            info = json.load(f)

        mtd_time = info['details']['mtd_time']
        mtd_times.append(mtd_time)

    idx = np.argsort(mtd_times).tolist()
    dirs = [dirs[i] for i in idx]

    return dirs


def make_subfolders(job_dir,
                    batch_size):

    dirs = [os.path.join(job_dir, i) for i in os.listdir(job_dir)]
    dirs = [i for i in dirs if os.path.isdir(i)]
    # order by metadynamics times to group molecules with similar run times
    # together
    dirs = sort_by_time(dirs)

    num_split = int(np.ceil(len(dirs) / batch_size))
    split_dirs = np.split(dirs, [batch_size * i for
                                 i in range(1, num_split)])
    split_dirs = [i.tolist() for i in split_dirs]
    split_str = "\n".join([" ".join(lst) for lst in split_dirs])

    return split_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_dir",
                        type=str)
    parser.add_argument("--batch_size",
                        type=int)
    args = parser.parse_args()
    split_str = make_subfolders(job_dir=args.job_dir,
                                batch_size=args.batch_size)
    print(split_str)


if __name__ == "__main__":
    main()
