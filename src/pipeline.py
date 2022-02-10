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

pipe = PipelineController('T4C pipeline', 't4c', '0.0.1')
pipe.set_default_execution_queue('services')

# TODO if dataset is already uploaded, don't go through subset creation
pipe.add_step(name='stage_data', base_task_project='t4c', base_task_name='subset_creation',
              pre_execute_callback = pre_execute_cb, post_execute_callback = post_execute_cb)
pipe.add_step(name='train', base_task_project='t4c', base_task_name='train_model',
              pre_execute_callback = pre_execute_cb, post_execute_callback = post_execute_cb)

#pipe.start_locally()

pipe.start()
print('done')
