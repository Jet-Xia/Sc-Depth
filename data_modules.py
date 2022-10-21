from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

import datasets.custom_transforms as custom_transforms
from config import get_training_size
from datasets.train_folders import TrainFolder
from datasets.validation_folders import ValidationSet


# DataLoader Module
# trainer.fit(Model, DataLoader)
class VideosDataModule(LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.training_size = get_training_size(hparams.dataset_name)

        # data loader
        # 训练预处理
        self.train_transform = custom_transforms.Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.RescaleTo(self.training_size),
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize()]
        )
        #
        self.valid_transform = custom_transforms.Compose([
            custom_transforms.RescaleTo(self.training_size),
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize()]
        )

    def prepare_data(self):
        pass

    #
    def setup(self, stage=None):

        # training dataset
        self.train_dataset = TrainFolder(
            self.hparams.hparams.dataset_dir,
            transform=self.train_transform,
            train=True,
            sequence_length=self.hparams.hparams.sequence_length,
            skip_frames=self.hparams.hparams.skip_frames,
            use_frame_index=self.hparams.hparams.use_frame_index
        )

        # validatioin dataset
        if self.hparams.hparams.val_mode == 'depth':
            self.val_dataset = ValidationSet(
                self.hparams.hparams.dataset_dir,
                transform=self.valid_transform,
                dataset=self.hparams.hparams.dataset_name
            )
        elif self.hparams.hparams.val_mode == 'photo':
            self.val_dataset = TrainFolder(
                self.hparams.hparams.dataset_dir,
                transform=self.valid_transform,
                train=False,
                sequence_length=self.hparams.hparams.sequence_length,
                skip_frames=self.hparams.hparams.skip_frames,
                use_frame_index=self.hparams.hparams.use_frame_index
            )
        else:
            print("wrong validation mode")

        print('{} samples found for training'.format(len(self.train_dataset)))
        print('{} samples found for validatioin'.format(len(self.val_dataset)))

    # training dataloader
    def train_dataloader(self):
        # 对整个 dataset 进行随机打乱顺序
        sampler = RandomSampler(self.train_dataset,
                                replacement=True,
                                num_samples=self.hparams.hparams.batch_size * self.hparams.hparams.epoch_size)
        return DataLoader(self.train_dataset,
                          sampler=sampler,
                          num_workers=4,
                          batch_size=self.hparams.hparams.batch_size,
                          pin_memory=True)

    # validatioin dataloader
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.hparams.batch_size,
                          pin_memory=True)
