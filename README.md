## General:

use the environment.yml with anaconda.

## Rerun experiments from the PermeabilityNets paper:

1.  Download the data and checkpoints from here:
`TODO`
 The data is split into a Training Validation and Testing set. Note that at least 128gb of RAM and a capable graphics card is required to run the full training!
    The evaluation should run on 16gb RAM and a GPU with at least 8gb of vRAM. 
    
2. Open ModelTrainerScripts.flowfront_to_fiber_STANDALONE.py. This is a version of our training script that is modified to work as a demo script. Only very little setup is needed:
   1. save_path should be a _Path()_ that specifies the output folder. Training and Eval results will be saved there.
   2. data_folder should be a _Path()_ that points to the folder containing the three data splits downloaded in step 1.
   3. model_chkpts should be a _Path()_ that points to the folder containing the four model checkpoints (from step 1.). Do not rename the checkpoints, since it will break the demo script!
   4. mode: choose between Conv2D, Conv3D, Transformer, or ConvLSTM.
   5. eval: set to False if you want to run the full training. For the ConvLSTM and Conv3D model this could take up to a week, depending on your ML-hardware. 
3.
    Run your configuration.

#### A Translation for the Code:
* ConvLSTM is called `FFTFF` in Models.flowfront_to_fiber_fraction_model.py
* Transformer is called `OptimusPrime` in Models.flowfront2PermTransformer.py
* Conv2D is called `FF2Perm_Baseline` in Models.flowfront2PermBaseline.py
* Conv3D is called `FF2Perm_3DConv` in Models.flowfront2PermBaseline.py

### Short Guide to the RTM-Predictions Data-Pipeline (regular use):

* Most of the work is done by the `ModelTrainer` class (defined in `Trainer.ModelTrainer.py`)
* `ModelTrainer` is the generic base class which takes all necessary parameters for any kind of training
* The training process is configured and started from an own dedicated script
* Examples for these dedicated scripts can be found in the model_trainer_*.py files in the root directory 

Basic principles of ModelTrainer:
* Data processing is currently done by the 'LoopingDataGenerator' (defined in `Pipeline.torch_datagenerator.py`) 
* `LoopingDataGenerator` takes a list of file paths as base paths
* The base paths are searched for .erfh5 files using the `data_gather_function` passed to the ModelTrainer 
* After gathering the .erfh5 files, the data from them is processed using the `data_processing_function` passed to the ModelTrainer. An example for processing is the extraction of all pressure sensor values 
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


