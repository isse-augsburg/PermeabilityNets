from pathlib import Path 
import torch
from torch.optim.lr_scheduler import ExponentialLR

import Resources.training as r
from Models.erfh5_ConvModel import S80Deconv2ToDrySpotEff
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Pipeline import torch_datagenerator as td
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params
from Utils.data_utils import handle_torch_caching
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import time 

if __name__ == "__main__":
    args = read_cmd_params()

    dl = DataloaderDryspots(sensor_indizes=((0, 1), (0, 1)))
    print("Created Dataloader.")
    load_and_save_path, data_loader_hash = handle_torch_caching(
    dl.get_flowfront_bool_dryspot, r.get_data_paths_base_0(), r.datasets_dryspots)

    generator = td.LoopingDataGenerator(
                r.get_data_paths_base_0(),
                get_filelist_within_folder_blacklisted,
                dl.get_sensor_bool_dryspot,
                batch_size=512,
                num_validation_samples=131072,
                num_test_samples=1048576,
                split_load_path=r.datasets_dryspots, 
                split_save_path=r.save_path,
                num_workers=75,
                cache_path=r.cache_path,
                cache_mode=td.CachingMode.Both,
                save_torch_dataset_path=load_and_save_path,
                load_torch_dataset_path=load_and_save_path,
                dont_care_num_samples=False,
                test_mode=False
            )
    print("Created Datagenerator")

    train_data = []
    train_labels = []

    for inputs, labels, _ in generator:

        train_data.extend(inputs.numpy())
        train_labels.append(labels.numpy())
    
    print("Loaded train data.")

    test_datagenerator = td.LoopingDataGenerator(
                    r.get_data_paths_base_0(),
                    get_filelist_within_folder_blacklisted,
                    dl.get_sensor_bool_dryspot,
                    batch_size=512,
                    num_validation_samples=131072,
                    num_test_samples=1048576,
                    split_load_path=r.datasets_dryspots, 
                    split_save_path=r.save_path,
                    num_workers=75,
                    cache_path=r.cache_path,
                    cache_mode=td.CachingMode.Both,
                    save_torch_dataset_path=load_and_save_path,
                    load_torch_dataset_path=load_and_save_path,
                    dont_care_num_samples=False,
                    test_mode=True
                )

    test_set = test_datagenerator.get_test_samples()
    test_data = [data.numpy() for (data, _, _) in test_set]
    test_labels = [label.numpy() for (_, label, _) in test_set]
    print("Loaded Test data.")

    classifiers = [
    ("Nearest Neighbors", KNeighborsClassifier(3)), # 5 worse then 3
    ("Decision Tree", DecisionTreeClassifier(max_depth=50)), # better then 5
    ("Random Forest", RandomForestClassifier(n_estimators=15)), # better 
    ("Neural Net", MLPClassifier(alpha=0.5, max_iter=5000)), # no difference
    # ("AdaBoost", AdaBoostClassifier()), 
    # ("Naive Bayes", GaussianNB()), 
    # ("QDA", QuadraticDiscriminantAnalysis()), 
    # ("Linear SVM", SVC(kernel="linear", C=0.125)),
    # ("RBF SVM", SVC(gamma=2, C=1)),
    # ("GaussianProcess", GaussianProcessClassifier(1.0 * RBF(1.0)))
    ]
    print("Created classifiers")

    for name, clf in classifiers:
        print(f"Training and evaluating {name}.")
        start_time = time.time()

        clf.fit(train_data, train_labels)
        preds = clf.predict(test_data)
        accuracy = metrics.accuracy_score(test_labels, preds)

        print(f"{name:18}: {round(accuracy * 100, 5)} % Accuracy")
        print(metrics.confusion_matrix(test_labels, preds))
        

        taken = round(time.time() - start_time, 2)
        print(f"Time taken: {taken} sec\n")

