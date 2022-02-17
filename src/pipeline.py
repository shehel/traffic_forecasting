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


pipe = PipelineController('T4C pipeline', 't4c', '0.0.2')
pipe.set_default_execution_queue('services')

pipe.add_parameter(
     name='data_path',
     description='path to raw t4c data',
     default='/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/'
)
pipe.add_step(name='set_env', base_task_project='t4c', base_task_name='create_env',
              parameter_override={'Args/data_dir':'${pipeline.data_path}'})
# TODO if dataset is already uploaded, don't go through subset creation
pipe.add_step(name='stage_data', base_task_project='t4c', base_task_name='subset_creation',
                    task_overrides={"script.version_num":"347e7e5c2941c48063ca9bd4b07d056fcd44123a","script.branch": "master"})
pipe.add_step(name='train', base_task_project='t4c', base_task_name='train_model')


pipe.start_locally()

#pipe.start()
print('done')
