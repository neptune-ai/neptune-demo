import neptune

# Select project
neptune.init(api_token='ANONYMOUS',
             project_qualified_name='shared/neptune-demo')

# Create experiment
neptune.create_experiment(name='bare_minimal_example')

# Log some metrics
for i in range(100):
    neptune.log_metric('loss', 0.95**i)
