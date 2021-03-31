## General:

use the environment.yml with anaconda.

## Rerun experiments from the PermeabilityNets paper:

1.  Download the data here: https://figshare.com/s/d403ed0cb9816507cb4c and the checkpoints here: https://figshare.com/s/3bce79cf4a47ab0986ae
 The data is split into a Training, Validation and Testing set. Note that at least 128 GB of RAM and a capable graphics card is required to run the full training!
    The evaluation should run on 16gb RAM and a GPU with at least 8 GB of vRAM. 
    
2. Open ModelTrainerScripts.flowfront_to_fiber_STANDALONE.py. This is a version of our training script that is modified to work as a demo script. Only very little setup is needed:
   1. save_path should be a _Path()_ that specifies the output folder. Training and Eval results will be saved there.
   2. data_folder should be a _Path()_ that points to the folder containing the three data splits downloaded in step 1.
   3. model_chkpts should be a _Path()_ that points to the folder containing the four model checkpoints (from step 1.). Do not rename the checkpoints, since it will break the demo script!
   4. mode: choose between Conv2D, Conv3D, Transformer, or ConvLSTM.
   5. eval: set to False if you want to run the full training. For the ConvLSTM and Conv3D model this could take up to a week, depending on your ML-hardware. 
3.
    Run your configuration.

#### A Translation for the Code:
* ConvLSTM is called `FFTFF` in Models.sensor_to_fiber_fraction_model.py
* Transformer is called `OptimusPrime` in Models.flowfront2PermTransformer.py
* Conv2D is called `FF2Perm_Baseline` in Models.flowfront2PermBaseline.py
* Conv3D is called `FF2Perm_3DConv` in Models.flowfront2PermBaseline.py

