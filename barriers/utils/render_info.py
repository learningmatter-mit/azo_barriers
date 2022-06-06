import os
import json
import argparse
import shutil

def make_info(script_dir,
              cwd):

    default_path = os.path.join(script_dir, 'default_details.json')
    with open(default_path, 'r') as f:
        info = json.load(f)

    info_path = os.path.join(cwd, 'job_info.json')
    with open(info_path, 'r') as f:
        update = json.load(f)

    info.update(update)
    shutil.move(info_path, info_path.replace("info", "info_input"))

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cwd',
                        help="Current directory",
                        type=str)
    parser.add_argument('--script_dir',
                        help="Location of job script",
                        type=str)

    args = parser.parse_args()
    make_info(cwd=args.cwd,
              script_dir=args.script_dir)


if __name__ == "__main__":
    main()
