from nff.utils.confgen import confs_and_save
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_file",
                        help="Path to JSON file with confgen information",
                        default='job_info.json')
    args = parser.parse_args()
    confs_and_save(config_path=args.info_file)


if __name__ == "__main__":
    main()
