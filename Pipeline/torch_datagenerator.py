import logging
import torch

from .Utils.torch_internal import FileDiscovery, FileSetIterable, CachingMode, SubSetGenerator
from .Utils.looping_strategies import LoopingStrategy, DataLoaderListLoopingStrategy, NoOpLoopingStrategy


class LoopingDataGenerator:
    """ An iterable for a batches of samples stored in files.

    Args:
        data_paths (list of paths): The data paths for gathering data
        gather_data (function): A callable that gathers files given a single root directory.
            data_gather.get_filelist_within_folder is usually used for this.
        load_data (function): A function that can load a list of samples given a filename 
            MUST return the following format:
            [(data_1, label_1), ... , (data_n, label_n)]
        batch_size (int): The batch size
        epochs (int): The number of epochs. The iteration will stop once epochs*batch_size samples where produced.
        num_validation_samples (int): The number of samples in the validation subset
        num_test_samples (int): The number of samples for the test subset
        split_load_path  (int): The directory to load validation and test set splits from
        split_save_path  (int): The directory to save validation and test set splits to
        num_workers (int): The number of worker processes for the dataloader. Defaults to 0 so that no additional
            processes are spawned.
        cache_path (Path): The cache directory for file lists and samples
        cache_mode (CachingMode): The cache mode. If set to FileLists, only lists of gathered files will be stored.
        looping_strategy (LoopingStrategy): The strategy for looping samples.
            Defaults to the DataLoaderListLoopingStrategy if more than one epoch is used,
            otherwise the NoOpLoopingStrategy will be used.
    """
    def __init__(self,
                 data_paths,
                 gather_data,
                 load_data,
                 batch_size=1,
                 epochs=1,
                 num_validation_samples=0,
                 num_test_samples=0,
                 split_load_path=None,
                 split_save_path=None,
                 num_workers=0,
                 cache_path=None,
                 cache_mode=CachingMode.Both,
                 looping_strategy: LoopingStrategy = None
                 ):
        self.epochs = epochs  # For compatibility with the MasterTrainer
        self.batch_size = batch_size  # For compatibility with the MasterTrainer
        self.num_workers = num_workers
        self.remaining_epochs = epochs
        self.store_samples = True
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.cache_mode = cache_mode
        self.logger = logging.getLogger(__name__)

        if looping_strategy is None:
            if epochs > 1:
                looping_strategy = DataLoaderListLoopingStrategy(batch_size)
            else:
                looping_strategy = NoOpLoopingStrategy()
        self.looping_strategy = looping_strategy
        self.logger.debug(f"Using {type(self.looping_strategy).__name__} for looping samples across epochs.")

        all_files = self._discover_files(data_paths, gather_data)
        self.logger.info("Generating validation and test data splits.")
        self.eval_set_generator = SubSetGenerator(load_data, "validation_set", num_validation_samples,
                                                  load_path=split_load_path, save_path=split_save_path)
        self.test_set_generator = SubSetGenerator(load_data, "test_set", num_test_samples,
                                                  load_path=split_load_path, save_path=split_save_path)
        remaining_files = self.eval_set_generator.prepare_subset(all_files)
        remaining_files = self.test_set_generator.prepare_subset(remaining_files)
        self.logger.info(f"{len(remaining_files)} files remain after splitting eval and test sets.")
        self.file_iterable = FileSetIterable(remaining_files, load_data,
                                             cache_path=cache_path, cache_mode=cache_mode)
        self.iterator = None

        self.logger.info("Data generator initialization is done.")

    def _discover_files(self, data_paths, gather_data):
        self.logger.info(f"Gathering files from {len(data_paths)} paths...")
        data_paths = [str(x) for x in data_paths]
        discovery = FileDiscovery(gather_data, cache_path=self.cache_path, cache_mode=self.cache_mode)
        paths = discovery.discover(data_paths)
        self.logger.debug(f"Gathered {len(paths)} files.")
        return paths

    def _create_initial_dataloader(self):
        # By choosing drop_last=False we may get up to num_workers*(batch_size-1) short batches in the first epoch.
        # The behaviour in the second depends on the used LoopingStrategy, but by default we will only see one short
        # sample in the following epochs
        self.iterator = iter(torch.utils.data.DataLoader(self.file_iterable, drop_last=False,
                                                         batch_size=self.batch_size, num_workers=self.num_workers))

    def __iter__(self):
        return self

    def __next__(self):
        """ Get the next batch of samples
        """
        if self.iterator is None:
            self._create_initial_dataloader()
        try:
            batch = next(self.iterator)
            if self.store_samples:
                batch = [e.clone() for e in batch]
                self.looping_strategy.store(batch)
        except StopIteration:
            self.remaining_epochs -= 1
            if self.remaining_epochs == 0:
                raise StopIteration
            self.logger.info(f"Starting epoch {self.epochs - self.remaining_epochs + 1}")
            self.store_samples = False
            self.iterator = self.looping_strategy.get_new_iterator()
            batch = next(self.iterator)
        return batch[0], batch[1]

    def get_validation_samples(self):
        """ Get the set of validation samples
        """
        return self.eval_set_generator.get_samples()

    def get_test_samples(self):
        """ Get the set of test samples
        """
        return self.test_set_generator.get_samples()

    def end_threads(self):
        # TODO: Dummy method for compatibility with the old pipeline. Remove this once the old pipeline
        # is removed
        pass

    def get_current_queue_length(self):
        # TODO: Dummy method for compatibility with the old pipeline. Remove this once the old pipeline
        # is removed
        return "unk"
