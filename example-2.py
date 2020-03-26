import neptune
import numpy as np

# Select project
neptune.init(api_token='ANONYMOUS',
             project_qualified_name='shared/neptune-demo')

# Define parameters
PARAMS = {'decay_factor': 1.0,
          'n_iterations': 117}

# Create experiment
neptune.create_experiment(name='quick_start_example',
                          params=PARAMS)

# Log some metrics
for i in range(1, PARAMS['n_iterations']):
    neptune.log_metric('iteration', i)
    neptune.log_metric('loss', PARAMS['decay_factor']/i**0.5)
    neptune.log_text('text_info', 'some value {}'.format(0.95*i**2))

# Log some images
for j in range(5):
    array = np.random.rand(10, 10, 3)*255
    array = np.repeat(array, 30, 0)
    array = np.repeat(array, 30, 1)
    neptune.log_image('mosaics', array)
