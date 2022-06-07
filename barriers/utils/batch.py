import os
import argparse


def render_tasks(cwd,
                 script_dir):

    tasks_text = ""

    for i in os.listdir(cwd):
        sub_dir = os.path.join(cwd, i)
        if not os.path.isdir(sub_dir):
            continue

        info_path = os.path.join(sub_dir, 'job_info.json')
        if not os.path.isfile(info_path):
            continue

        tasks_text += "cd %s && bash job.sh && cd - \n" % sub_dir

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
    render_tasks(cwd=args.cwd,
                 script_dir=args.script_dir)


if __name__ == "__main__":
    main()
