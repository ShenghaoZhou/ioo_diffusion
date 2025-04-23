import lightning.pytorch as pl
from airimu_datasets import SeqeuncesDataset, collate_fcs, collate_with_padding, collate_with_padding_ori9
from pyhocon import ConfigFactory
from torch.utils.data import DataLoader


class AirIMUData(pl.LightningDataModule):
    def __init__(self, cfg, airimu_cfg_path):
        super().__init__()
        self.cfg = cfg
        self.airimu_conf = ConfigFactory.parse_file(airimu_cfg_path)
        # original collate function, working with batch_size = 1 only
        # self.collate_fn = collate_fcs[self.airimu_conf.dataset.collate]

        # my version, plain padding scheme (reasonable to do), working with any batch_size
        self.collate_fn = collate_with_padding

        # my version to match the original collate function padding scheme, working with any batch_size
        # self.collate_fn = collate_with_padding_ori9

    def setup(self, stage=None):
        train_conf = self.airimu_conf.dataset.train
        train_conf.data_list[0]["window_size"] = self.cfg.seqlen
        # FIXME: overlap by 50%, hope to get better prediction consistency
        train_conf.data_list[0]["step_size"] = self.cfg.seqlen // self.cfg.overlap_factor
        self.train_dataset = SeqeuncesDataset(train_conf)

        test_conf = self.airimu_conf.dataset.inference
        # overwrite window size and step size to desired value
        test_conf.data_list[0]["window_size"] = self.cfg.seqlen
        test_conf.data_list[0]["step_size"] = self.cfg.seqlen
        test_conf["mode"] = "infevaluate"

        # FIXME: there are more changes in the original inference.py. We omit them because they look useless
        self.test_dataset = SeqeuncesDataset(test_conf)
        # self.airimu_test_conf = test_conf

    def train_dataloader(self):
        # by default, the training will use collate_fn["base"], which will keep three different groups of the full dict
        # pytorch by default will perform dict merging by key, resulting in a big dict
        # the difference is minor (doen't affect the loaded data), and we don't bother using the custom collate_fn
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=2048, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=8)  # drop_last is False by default
        # return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False)
