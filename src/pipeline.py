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

# TODO if dataset is already uploaded, don't go through subset creation
pipe.add_step(name='stage_data', base_task_project='t4c', base_task_name='subset_creation',
                    task_overrides={"script.version_num":"3c1b660826c90a55a6c246c9f5ca18982ea2acff","script.branch": "master"})
pipe.add_step(name='train', base_task_project='t4c', base_task_name='train_model',)
              #task_overrides={"script.version_num":"3c1b660826c90a55a6c246c9f5ca18982ea2acff","script.branch": "master"})
# YAML override: parameter_override={'Args/overrides': '[the_hydra_key={}]'.format(a_new_value)})
pipe.start_locally()

#pipe.start()
print('done')
