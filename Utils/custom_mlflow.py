import mlflow

logging = True

def log_param(key, value):
    if logging:
        mlflow.log_param(key, value)

def log_metric(key, value, step=None):
    if logging:
        mlflow.log_metric(key, value, step=None)

def set_tag(key, value):
    if logging:
        mlflow.set_tag(key, value)

def log_artifacts(local_dir, artifact_path=None):
    if logging:
        mlflow.log_artifacts(local_dir, artifact_path=None)

def set_tracking_uri(uri):
    if logging:
        mlflow.set_tracking_uri(uri)

def set_experiment(experiment_name):
    if logging:
        mlflow.set_experiment(experiment_name)
