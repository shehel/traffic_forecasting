import sys
import argparse
import os

from clearml import Task
from clearml.automation import PipelineController

def pre_execute_cb(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print('Cloning Task id={} with parameters: {}'.format(a_node.base_task_id, current_param_override))
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True

def post_execute_cb(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print('Completed Task id={}'.format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                       required=True, help="absolute path of where the raw T4C dataset resides")
    args = parser.parse_args(args)
    cwd = os.getcwd()

    env_file = open(".env","w+")
    env_file.write("export PROJECT_ROOT=\""+cwd+"\"\n")
    env_file.write("export DATA_PATH=\""+args.data_dir+"\"")
    env_file.close()

    pipe = PipelineController('T4C pipeline', 't4c', '0.0.1')
    pipe.set_default_execution_queue('services')

    # TODO if dataset is already uploaded, don't go through subset creation
    pipe.add_step(name='stage_data', base_task_project='t4c', base_task_name='subset_creation')
    pipe.add_step(name='train', base_task_project='t4c', base_task_name='train_model')


    pipe.start_locally()

    #pipe.start()
    print('done')

if __name__ == "__main__":
    main(sys.argv[1:])
