import copy
import os
import shutil
import json
import argparse


def load_info(cwd):
    path = os.path.join(cwd, 'job_info.json')
    with open(path, 'r') as f:
        info = json.load(f)
    return info


def info_to_sub_dirs(cwd,
                     script_dir):

    info = load_info(cwd)
    keys = ['nxyz_list', 'xyz_list', 'coord_list',
            'nxyz', 'xyz', 'coords']
    found = False
    for key in keys:
        if key in info:
            use_key = key
            found = True
            break

    if not found:
        key_str = ", ".join(keys)
        msg = "None of the following keys could be found: %s" % key_str
        raise Exception(msg)

    for key in keys:
        if key != use_key and key in info:
            info.pop(key)

    val_list = info[use_key]
    msg = "You seem to have provided one set of coordinates, not multiple"
    if 'coords' in use_key:
        assert type(val_list[0]) is list, msg
    elif 'xyz' in use_key:
        assert type(val_list[0][0]) is list, msg

    tasks_text = ""
    print("Creating sub-directories for each set of coordinates...")

    for i, val in enumerate(val_list):
        new_info = copy.deepcopy(info)
        if 'xyz' in use_key:
            single_key = 'nxyz'
        else:
            single_key = 'coords'

        new_info.pop(use_key)
        new_info[single_key] = val

        new_dir = os.path.join(cwd, str(i))
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

        shutil.copy(os.path.join(script_dir, 'job.sh'),
                    os.path.join(new_dir, 'job.sh'))

        info_path = os.path.join(new_dir, 'job_info.json')
        with open(info_path, 'w') as f:
            json.dump(new_info, f, indent=4)

        tasks_text += "cd %s && bash job.sh && cd - \n" % new_dir

    tasks_file = os.path.join(cwd, 'tasks.txt')
    with open(tasks_file, "w") as f:
        f.write(tasks_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cwd',
                        help="Current directory",
                        type=str)
    parser.add_argument('--script_dir',
                        help="Location of job script",
                        type=str)

    args = parser.parse_args()
    info_to_sub_dirs(cwd=args.cwd,
                     script_dir=args.script_dir)


if __name__ == "__main__":
    main()
