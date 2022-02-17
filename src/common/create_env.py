import os
import sys
import argparse

from clearml import Task


def main(args):
    task = Task.init(project_name='t4c', task_name='create_env')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                       required=True, help="absolute path of where the raw T4C dataset resides")

    args = parser.parse_args(args)
    cwd = os.getcwd()

    env_file = open(".env","w+")
    env_file.write("export PROJECT_ROOT=\""+cwd+"\"\n")
    env_file.write("export DATA_PATH=\""+args.data_dir+"\"")
    env_file.close()
    print (cwd)
    print (os.listdir(cwd))

if __name__ == "__main__":
    main(sys.argv[1:])
