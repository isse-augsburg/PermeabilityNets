import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from pathlib import Path


def plot_confusion_matrix(cm, class_names, norm='', show_plot=False, save_as=None):
    """
    Plots a confusion matrix of arbitrary size

    Args:
        cm (2D numpy.ndarray [int]): array that represents the confusion matrix to plot (unnormalized)
        class_names (list [str]): list of corresponding class names
        norm: type of normalization to apply {'by_class', 'overall', default: no normalization}
        show_plot: if True, display plot in a window during runtime
        save_as (pathlib.Path or str): full path, including filename and type (e.g. '/cfs/example/confmat.png')

    """
    assert cm.shape[0] == cm.shape[1] and cm.shape[0] == len(class_names)
    assert norm in ['', 'by_class', 'overall']

    plt.rcParams['figure.constrained_layout.use'] = True
    fig = plt.figure(figsize=(len(class_names) + 1, len(class_names) + 1), dpi=150)

    if norm == 'by_class':
        cm = np.around(normalize(cm, norm='l1', axis=1), decimals=2)
    elif norm == 'overall':
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

    if show_plot:
        plt.show()

    if save_as is not None:
        Path(save_as).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_as)

    return fig


def plot_sample_eval(images: list,
                     sub_titles=None,
                     main_title=None,
                     vmin=None, vmax=None,
                     label_str=None, pred_str=None,
                     additional_info=None,
                     show_plot=False, save_as=None):
    """
    Plots one or multiple images in a row, including titles and additional information, if given.
    Recommended to use for visualising network input, prediction, label etc. of a data sample or time step

    Args:
        images (list[2D numpy.ndarray]): Images to display in the plot, e.g. sensor frames, flowfronts etc.
        sub_titles (list[str]): list of titles that will be displayed above the corresponding image. Length should match
                                the number of images
        main_title (str): the main title displayed at the top
        vmin (list[float or int]): set the min value for each subplot manually (useful for time series plots).
                                   Length should match the number of images
        vmax (list[float or int]): set the max value for each subplot manually (useful for time series plots).
                                   Length should match the number of images
        label_str: Label as a string (useful if label is a class, not an image)
        pred_str: Prediction as a string (useful if prediction is a class, not an image)
        additional_info (list[str]): List of strings that will be displayed at the bottom of the plot. Each list entry
                                     is put in a new row.
        show_plot: if True, the plot will be shown in a window during runtime
        save_as (pathlib.Path or str): full path, including filename and type (e.g. '/cfs/example/output.png')

    """
    assert bool(images)
    assert sub_titles is None or len(sub_titles) == len(images)
    assert vmin is None or len(vmin) == len(images)
    assert vmin is None or len(vmin) == len(images)

    plt.rcParams['figure.constrained_layout.use'] = True

    # set up figure size and basic structure
    ratio = images[0].shape[0] / images[0].shape[1]
    base_size = 4
    text_space = 0.35 if main_title is not None else 0
    text_space += 0.35 if label_str is not None else 0
    text_space += 0.35 if pred_str is not None else 0
    text_space += 0.35 * len(additional_info) if additional_info is not None else 0
    figsize = (base_size * len(images), base_size * ratio + text_space)
    fig, axs = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axs = [axs]

    if main_title is not None:
        fig.suptitle(main_title)

    for i, img in enumerate(images):
        axs[i].imshow(img, vmin=None if vmin is None else vmin[i], vmax=None if vmax is None else vmax[i])
        axs[i].set(xticks=[], yticks=[], title=None if sub_titles is None else sub_titles[i])

    text = ""
    color = 'black'

    if label_str is not None:
        text += f"{'Label: ':8}{label_str}"
    if label_str is not None and pred_str is not None:
        color = 'green' if label_str == pred_str else 'red'
        text += '\n'
    if pred_str is not None:
        text += f"{'Pred: ':8}{pred_str}"

    if additional_info is not None:
        for info in additional_info:
            text += f"\n{info}"

    plt.figtext(0.01, 0.01, text, c=color, ha='left')

    if show_plot:
        plt.show()

    if save_as is not None:
        Path(save_as).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_as)

    return fig


if __name__ == "__main__":
    test_sensors = np.random.rand(38, 30)
    test_flowfront = np.random.rand(143, 111)
    test_no3 = np.random.rand(143, 111)
    # print(test_sensors)
    aux_info = [f"Original num of states: 475 (250 with dryspot info)",
                f"Original num of states: 475 (250 with dryspot info)",
                f"Original num of states: 475 (250 with dryspot info)"]
    imgs = [test_sensors, test_flowfront, test_no3]
    titles = ['Sensor values', 'Flowfront', 'Nochmal was']
    title = 'Test title'
    plot_sample_eval(imgs, titles, title, label_str="OK", pred_str="OK", additional_info=aux_info, show_plot=True)
    plot_sample_eval(imgs, titles, label_str="OK", pred_str="OK", additional_info=aux_info, show_plot=True)
    plot_sample_eval([test_sensors], [titles[1]], title, label_str="OK", show_plot=True)
    plot_sample_eval([test_sensors], [titles[1]], label_str="OK", show_plot=True)
    # plot.savefig("testplot.png", bbox_inches='tight')
