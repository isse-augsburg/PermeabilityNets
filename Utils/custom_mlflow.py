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


# primarily used to set run_name
def start_run(run_id=None, experiment_id=None, run_name=None, nested=False):
    if logging:
        mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=nested)


def end_run():
    if logging:
        mlflow.end_run()


def get_artifact_uri():
    if logging:
        return mlflow.get_artifact_uri()
    return None
