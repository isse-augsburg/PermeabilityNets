## General:
To run the code, use Docker:
```
cd Docker
docker build -t "pytorch_extended:20.02"
docker run ...
```
Alternatively, use the environment.yml with anaconda.

## Rerun experiments from FlowFrontNet paper:

1.  Download the data and checkpoints from here:
https://figshare.com/s/dde2f78958173c23aee4.
There are two big Zip files: SensorToDryspot, SensorToFlowFront.
To recreate the experiments from the paper, we need both.
    * The SensorToDryspot dataset can be used for:
        * Feedforward baseline
        * Finetuning FlowFrontNet with a pretrained Deconv / Conv 
    * The SensorToFlowFront dataset can be used to train the Deconv / Conv Network to produce FlowFrontImages

2. Unzip those files in a certain `data_path`: `SensorToFlowFront` and `SensorToDrySpot`

3. Start Trainings:
    * Start the following script for 1140 sensors to flowfront:\
    `python3 -u ModelTrainerScripts.sensor_1140_to_flow.py --demo data_path/SensorToFlowFront`

    * To use the fine-tuned model for binary classification:\
    `python3 -u ModelTrainerScripts.sensor_1140_to_dryspot.py --demo data_path/SensorToDrySpot --checkpoint_path checkpoint_path`

    * For the baseline, run:\
    `python3 -u ModelTrainerScripts.sensor_1140_dryspot_end_to_end_dense.py --demo data_path/SensorToDrySpot`

4. Evaluation:
    * Start the following script for 1140 sensors to flowfront:\
    `python3 -u ModelTrainerScripts.sensor_1140_to_flow.py --demo data_path/SensorToFlowFront--eval eval_output_path --checkpoint_path checkpoint_path`
    
    * To use the fine-tuned model for binary classification:\
    `python3 -u ModelTrainerScripts.sensor_1140_to_dryspot.py --demo data_path/SensorToDrySpot --eval eval_output_path --checkpoint_path checkpoint_path`
    
    * For the baseline, run:\
    `python3 -u ModelTrainerScripts.sensor_1140_dryspot_end_to_end_dense.py --demo data_path/SensorToDrySpot --eval eval_output_path --checkpoint_path checkpoint_path`

Caution: New Folders with logs, tensorboard files etc. will be created in the directory of the Datasets, corresponding to the task: SensorToFlowFront or SensorToDryspot.
For the trainings and evaluations with 80 and 20 sensors use the respective `ModelTrainerScripts.sensor_*_...` scripts.

### Short Guide to the RTM-Predictions Data-Pipeline (regular use):

Note: A complete documentation can be found in `/cfs/share/cache/website/html/index.html` or http://137.250.170.59:8000/ or
https://rtm-predictions.readthedocs.io/en/latest/index.html (which might be not as up to date es the others).

* Most of the work is done by the `ModelTrainer` class (defined in `Trainer.ModelTrainer.py`)
* `ModelTrainer` is the generic base class which takes all necessary parameters for any kind of training
* The training process is configured and started from an own dedicated script
* Examples for these dedicated scripts can be found in the model_trainer_*.py files in the root directory 

Basic principles of ModelTrainer:
* Data processing is currently done by the 'LoopingDataGenerator' (defined in `Pipeline.torch_datagenerator.py`) 
* `LoopingDataGenerator` takes a list of file paths as base paths
* The base paths are searched for .erfh5 files using the `data_gather_function` passed to the ModelTrainer 
* After gathering the .erfh5 files, the data from these is processed using the `data_processing_function` passed to the ModelTrainer. An example for processing is the extraction of all pressure sensor values 
* Additional work such as creating batches and shuffling data is done automatically
* The training process is implemented in ModelTrainer. This includes validation steps during training and testing on a dedicated test set after training
 
Steps for using the ModelTrainer in your script: 
* For data processing you need two functions: 
    1. `data_gather_function`, a function for collecting the paths to the files from a root directory 
    2. `data_processing_function,` a function that extracts the data from the collected filepaths. **Must** return data in following format: `[(instance_1, label_1), ... , (instance_n, label_n)]`
    (examples for both functions can be found in the `Pipeline.data_loader_*.py` and `Pipeline.data_gather.py` files)
* Define a PyTorch model 
* Instantiate the ModelTrainer: `mt = ModelTrainer( ... )`. Pass all necessary arguments for your task. Important: You have to pass your model using `lambda: YourModel()`
* Train your model using `mt.start_training()`. No additional parameters need to be passed if you have configured the ModelTrainer correctly
* Testing using a dedicated test set can be done using `mt.inference_on_test_set( ... )`


### Usage of data chunks

* Set `dataset_split_path` parameter of `ModelTrainer` correctly, either directly or in `Resources.training`
* Pass `torch_datasets_chunk_size` with the correct data chunk size to `ModelTrainer`
* The correct chunks sizes for our pre-stored datasts are: sensor to dryspot -> 300 000, sensor to flowfront -> 75 000
* If you want to exclude some datachunks from loading, you either have to change the code of `Pipeline/TorchDataGeneratorUtils/torch_internal.load_data_chunks` or remove any files you don't want to load from the directory.

