#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from clearml import Task, Logger




task = Task.init(project_name="t4c_eval", task_name="Model img test")
logger = task.get_logger()
fig = plt.figure(figsize=(10, 7))
m = np.random.randn(496,448)
# setting values to rows and column variables
rows = 2
columns = 2

# reading images

logger.current_logger().report_image("image", "image float", iteration=0, image=m)
logger.current_logger().report_image("image", "image float", iteration=0, image=m)
logger.current_logger().report_image("image", "image float", iteration=1, image=m)
