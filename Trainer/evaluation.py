import itertools
import logging
import os
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path
from math import isnan

import matplotlib.pyplot as plt
import numpy as np
import pandas
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from torch.utils.tensorboard import SummaryWriter

from Utils.dicts.sensor_dicts import sensor_shape
from Utils.custom_mlflow import log_metric, get_artifact_uri

""" 
>>>> PLEASE NOTE: <<<<
Evaluation classes must provide three functions even if not all of them have functionality: 

* commit(output, label, inputs, aux): updates the evaluation state
* print_metrics(): prints a set of application-specific print_metrics
* reset: Resets the internal metrics of an evaluator, e.g. after a evaluation loop is finished.  

They have to be given a save_path, where the results are going to be stored.
"""


class Evaluator:
    def __init__(self):
        pass

    def commit(self, *args, **kwargs):
        pass

    def print_metrics(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass


def pixel_wise_loss_multi_input_single_label(input, target):
    loss = 0
    for el in input:
        out = el - target
        # out = out * weights.expand_as(out)
        loss += out.sum(0)
    return loss


def plot_predictions_and_label(input, target, _str):
    if os.name == "nt":
        debug_path = Path(r"X:\s\t\stiebesi\code\debug\overfit")
    else:
        debug_path = Path("/cfs/home/s/t/stiebesi/code/debug/overfit/")
    (debug_path / "predict").mkdir(parents=True, exist_ok=True)

    x = input.reshape(input.shape[0], 155, 155)
    x = x * 255
    with Pool() as p:
        p.map(
            partial(save_img, debug_path / "predict", _str, x),
            range(0, input.shape[0], 1),
        )
    y = target.reshape(target.shape[0], 155, 155)
    y = y * 255
    im = Image.fromarray(np.asarray(y[0]))
    path = debug_path / "label"
    path.mkdir(parents=True, exist_ok=True)
    file = f"{_str}.png"
    im.convert("RGB").save(path / file)
    im.close()


def save_img(path, _str, x, index):
    try:
        im = Image.fromarray(np.asarray(x[index]))
        file = f"{_str}_{index}.png"
        im.convert("RGB").save(path / file)
        im.close()
    except KeyError:
        logger = logging.getLogger(__name__)
        logger.error("ERROR: save_img")


class SensorToFlowfrontEvaluator(Evaluator):
    def __init__(self, save_path: Path = None,
                 sensors_shape=(38, 30),
                 skip_images=True,
                 print_n_images=-1,
                 ignore_inp=False):
        super().__init__()
        self.num = 0
        self.ignore_inp = ignore_inp
        self.save_path = save_path
        self.skip_images = skip_images
        if save_path is not None:
            self.im_save_path = save_path / "images"
            if not self.skip_images:
                self.im_save_path.mkdir(parents=True, exist_ok=True)
        self.sensors_shape = sensors_shape
        self.print_n_images = print_n_images

    def commit(self, net_output, label, inputs, aux, *args):
        if self.skip_images:
            return

        for sample in range(net_output.size()[0]):
            if self.num == self.print_n_images:
                break
            a = net_output[sample].numpy()
            a = np.squeeze(a)
            b = label[sample].numpy()
            b = np.squeeze(b)

            plt.imsave(self.im_save_path / Path(str(self.num) + "out.jpg"), a, vmin=0, vmax=1)
            plt.imsave(self.im_save_path / Path(str(self.num) + "lab.jpg"), b, vmin=0, vmax=1)
            if not self.ignore_inp:
                c = inputs[sample].numpy()
                c = np.squeeze(c)
                c = c.reshape(self.sensors_shape[0], self.sensors_shape[1])
                plt.imsave(self.im_save_path / Path(str(self.num) + "inp.jpg"), c)

            self.num += 1
        pass

    def print_metrics(self, step_count):
        pass

    def reset(self):
        self.num = 0
        pass


class BinaryClassificationEvaluator(Evaluator):
    """Evaluator specifically for binary classification. Calculates common metrics and a confusion matrix.
    """

    def __init__(self, save_path: Path = None,
                 skip_images=True,
                 with_text_overlay=False,
                 advanced_eval=False,
                 max_epochs=-1,
                 data_loader=None):
        super().__init__()
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0
        self.accuracy, self.balanced_accuracy, self.precision, self.recall, self.specificity = 0, 0, 0, 0, 0
        self.confusion_matrix = np.zeros((2, 2), dtype=int)
        self.evaluated_samples_epoch = 0
        self.epoch = 0
        self.max_epochs = max_epochs
        self.class_names = ["OK", "Not OK"]
        self.save_path = save_path
        self.skip_images = skip_images
        self.data_loader = data_loader
        if not self.skip_images:
            # when running TensorBoard, use '--samples_per_plugin images=100' to see all frames in slider
            self.writer = SummaryWriter(log_dir=Path(get_artifact_uri()))
            if save_path is not None:
                self.im_save_path = save_path / "images"
                self.im_save_path.mkdir(parents=True, exist_ok=True)
        self.num = 0
        self.with_text_overlay = with_text_overlay
        plt.set_loglevel('warning')
        if advanced_eval:
            self.origin_tracker = {}

    def commit(self, net_output, label, inputs, aux, *args):
        """Updates the confusion matrix.

        Args:
            net_output: predictions of the model
            label: associated labels
        """
        net_output = net_output.numpy()
        invalid = np.argwhere(np.isnan(net_output[:, 0]))
        if invalid.size > 0:
            invalid = np.reshape(invalid, (invalid.shape[0]))
            net_output = np.delete(net_output, invalid, 0)
            label = np.delete(label, invalid)

        predictions = np.around(net_output[:, 0])
        if hasattr(self, "origin_tracker"):
            for i, aux_sample in enumerate(aux):
                if aux_sample["sourcefile"] not in self.origin_tracker.keys():
                    self.origin_tracker[aux_sample["sourcefile"]] = {}
                self.origin_tracker[aux_sample["sourcefile"]][int(aux_sample['ix'])] = \
                    (net_output.flatten()[i], float(label[i]))

        self.confusion_matrix = np.add(self.confusion_matrix, confusion_matrix(label, predictions, labels=[0, 1]))

        if not self.skip_images:
            inputs = inputs.numpy()
            label = label.numpy().squeeze()

            # run-level classification (inputs has shape [#validation_samples, #frames, #sensors])
            if len(list(inputs.shape)) == 3 and self.epoch == self.max_epochs - 1:

                false_samples = np.where(predictions != label)
                num_sensors = inputs[0, 0].shape[0]

                logger = logging.getLogger(__name__)
                logger.info(f"False Samples: {false_samples[0]}")
                plt_vmax = np.amax(inputs)
                logger.info(f"Batch plt_vmax: {plt_vmax}")

                for sample_idx in range(min(25, len(list(predictions)))):  # false_samples[0]:
                    sample = inputs[sample_idx]
                    sample = np.squeeze(sample)
                    aux_sample = aux[sample_idx] if aux else {}
                    flowfronts = self.data_loader.load_aux_info_only(aux_sample['sourcefile'],
                                                                     aux_sample['single_state_indices'])
                    for i, frame in enumerate(sample):
                        frame = frame.reshape(sensor_shape[str(num_sensors)][0], sensor_shape[str(num_sensors)][1])
                        label_str = self.class_names[int(label[sample_idx])]
                        pred_str = self.class_names[int(predictions[sample_idx])]
                        aux_info = self.__get_aux_info(aux_sample, i)

                        frame_plot = self.plot_sensor_frame(frame, label_str, pred_str, aux_info, flowfronts[i], 0,
                                                            plt_vmax)
                        self.writer.add_figure(f"FalseClassified/{sample_idx + self.evaluated_samples_epoch}",
                                               frame_plot, i + 1)
                        plt.close(frame_plot)

            # frame-level classification (inputs has shape [#validation_samples, #sensors])
            if len(list(inputs.shape)) == 2:
                for sample in range(predictions.size):
                    c = inputs[sample]
                    c = np.squeeze(c)
                    c = c.reshape(143, 111)
                    ipred = int(predictions[sample])
                    ilabel = int(label[sample])
                    if self.with_text_overlay:
                        fig = plt.figure(figsize=(2, 1.55))
                        ax = fig.add_subplot(111)
                        ax.text(45., 75., f'Label={ilabel}\nPred={ipred}', c='red' if ipred != ilabel else 'green')
                        ax.imshow(c)
                        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        plt.axis("off")
                        plt.tight_layout()
                        plt.savefig(self.im_save_path / f"{self.num}-pred_{ipred}_label_{ilabel}.jpg",
                                    bbox_inches=extent)
                    else:
                        plt.imsave(
                            self.im_save_path / f"{self.num}-pred_{predictions[sample]}_label_{label[sample]}.jpg",
                            c)

        self.num += predictions.size
        self.evaluated_samples_epoch += predictions.size

    def print_metrics(self, step_count=0):
        """Prints and logs the counts of True/False Positives and True/False Negatives, (Balanced) Accuracy, Precision,
        Recall, Specificity and the confusion matrix.
        """
        self.__update_metrics()

        # Logger
        logger = logging.getLogger(__name__)
        logger.info(f"True positives: {self.tp}, False positives: {self.fp}, True negatives: {self.tn}, "
                    f"False negatives: {self.fn}")
        logger.info(f"Accuracy: {self.accuracy:7.4f}, Balanced Accuracy: {self.balanced_accuracy:7.4f}, "
                    f"Precision: {self.precision:7.4f}, Recall: {self.recall:7.4f}, "
                    f"Specificity: {self.specificity:7.4f}")
        df = pandas.DataFrame(self.confusion_matrix, columns=[0, 1], index=[0, 1])
        df = df.rename_axis('Pred', axis=0).rename_axis('True', axis=1)
        logger.info(f'Confusion matrix:\n{df}')

        # MLflow
        log_metric("Validation/Accuracy", self.accuracy, step_count)
        log_metric("Validation/Balanced_Accuracy", self.balanced_accuracy, step_count)
        log_metric("Validation/Precision", self.precision, step_count)
        log_metric("Validation/Recall", self.recall, step_count)
        log_metric("Validation/Specificity", self.specificity, step_count)

        log_metric("Confusion_Matrix/TN", self.tn, step_count)
        log_metric("Confusion_Matrix/FP", self.fp, step_count)
        log_metric("Confusion_Matrix/FN", self.fn, step_count)
        log_metric("Confusion_Matrix/TP", self.tp, step_count)

        # Confusion matrix plots for MLflow
        if get_artifact_uri() is not None:
            base_dir = Path(get_artifact_uri()) / "confusion_matrix"
            cm_types = ['absolute', 'normalized_overall', 'normalized_by_class']
            for cm_type in cm_types:
                cm_plot = self.__plot_confusion_matrix(self.confusion_matrix, self.class_names, cm_type)
                base_dir.joinpath(cm_type).mkdir(parents=True, exist_ok=True)
                cm_plot.savefig(base_dir / cm_type / f"epoch_{self.epoch:02}.png")
                plt.close(cm_plot)

    def __update_metrics(self):
        self.tn = self.confusion_matrix[0, 0]
        self.fp = self.confusion_matrix[0, 1]
        self.fn = self.confusion_matrix[1, 0]
        self.tp = self.confusion_matrix[1, 1]
        self.accuracy = (self.tp + self.tn) / max((self.tp + self.tn + self.fp + self.fn), 1e-8)
        self.precision = self.tp / max((self.tp + self.fp), 1e-8)
        self.recall = self.tp / max((self.tp + self.fn), 1e-8)
        self.specificity = self.tn / max((self.tn + self.fp), 1e-8)
        self.balanced_accuracy = (self.recall + self.specificity) / 2

    def reset(self):
        """Resets the internal counters for the next evaluation loop.
        """
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0
        self.confusion_matrix = np.zeros((2, 2), dtype=int)
        self.evaluated_samples_epoch = 0
        self.epoch += 1

    def __get_aux_info(self, aux, idx):
        aux_info = []
        if aux:
            framelabel = aux['framelabel'][idx]
            aux_info.append(f"Framelabel: {'-' if isnan(framelabel) else self.class_names[framelabel]}")
            aux_info.append(f"Original frame idx: {aux['original_frame_idx'][idx]:3} "
                            f"(of {aux['original_num_multi_states']})")
            aux_info.append(f"Sensors filled: {aux['percent_of_sensors_filled'][idx] * 100:.2f}%")
            aux_info.append(f"{aux['sourcefile']}")
        return aux_info

    @staticmethod
    def __plot_confusion_matrix(cm, class_names, norm=''):
        plt.rcParams['figure.constrained_layout.use'] = True
        fig = plt.figure(figsize=(len(class_names) + 1, len(class_names) + 1), dpi=150)

        if norm == 'normalized_by_class':
            cm = np.around(normalize(cm, norm='l1', axis=1), decimals=2)
        elif norm == 'normalized_overall':
            cm = np.around(cm / max(cm.sum(), 1e-8), decimals=2)

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=np.sum(cm, 1).max())
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # Use white text if squares are dark; otherwise black
        threshold = 0.5 * np.sum(cm, 1).max()
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        return fig

    @staticmethod
    def plot_sensor_frame(frame_sensors,
                          label=None, prediction=None,
                          additional_info=None, frame_flowfront=None,
                          vmin=None, vmax=None):
        # TODO Doc
        plt.rcParams['figure.constrained_layout.use'] = True

        ratio = frame_sensors.shape[0] / frame_sensors.shape[1]
        base_size = 4
        text_space = 0.7 if label is not None or prediction is not None else 0
        text_space += 0.35 * len(additional_info) if additional_info is not None else 0
        num_plots = 2 if frame_flowfront is not None else 1
        figsize = (base_size * num_plots, base_size * ratio + text_space)
        fig, axs = plt.subplots(1, num_plots, figsize=figsize)

        ax1 = axs[0] if num_plots == 2 else axs
        ax1.imshow(frame_sensors, vmin=vmin, vmax=vmax)
        ax1.set(xticks=[], yticks=[], title='Sensor values')

        if num_plots == 2:
            ax2 = axs[1]
            ax2.imshow(frame_flowfront)
            ax2.set(xticks=[], yticks=[], title='Flowfront (label)')

        text = ""
        color = 'black'

        if label is not None and prediction is not None:
            color = 'green' if label == prediction else 'red'
            text = f"{'Label: ':8}{label}\n{'Pred: ':8}{prediction}"
        elif label is not None:
            text = f"{'Label: ':8}{label}"
        elif prediction is not None:
            text = f"{'Pred: ':8}{prediction}"

        if additional_info is not None:
            for info in additional_info:
                text += f"\n{info}"

        plt.figtext(0.01, 0.01, text, c=color, ha='left')

        return fig


if __name__ == "__main__":
    test_sensors = np.random.rand(38, 30)
    test_flowfront = np.random.rand(143, 111)
    # print(test_sensors)
    aux_info = []
    aux_info.append(f"Original num of states: 475 (250 with dryspot info)")
    aux_info.append(f"Original num of states: 475 (250 with dryspot info)")
    plot = BinaryClassificationEvaluator.plot_sensor_frame(test_sensors, "OK", "OK", aux_info, test_flowfront)
    # plot.savefig("testplot.png", bbox_inches='tight')
