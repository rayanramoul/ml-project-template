import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split


# Custom PyTorch Dataset wrapping a Polars DataFrame
class PolarsDataset(Dataset):
    def __init__(self, df: pl.DataFrame, output_column: str):
        self.df = df
        self.output_column = output_column

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df[idx]
        features = torch.tensor([val for col, val in row.items() if col != self.output_column], dtype=torch.float32)
        label = torch.tensor(row[self.output_column], dtype=torch.long)
        return features, label


# PyTorch Lightning DataModule for loading dataset using Polars
class PolarsDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for loading dataset using Polars."""

    def __init__(
        self, data_path: str, output_column: str, batch_size: int = 32, num_workers: int = 0, test_size: float = 0.2
    ) -> None:
        """Initialize the PolarsDataModule.

        Args:
            data_path: Path to the dataset.
            output_column: Column name that contains the labels.
            batch_size: Batch size for the dataloaders.
            num_workers: Number of workers for the dataloaders.
            test_size: Fraction of the dataset to be used for validation.
        """
        super().__init__()
        self.data_path = data_path
        self.output_column = output_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.df = None  # Will hold the loaded Polars DataFrame

    def setup(self, stage: str = "") -> None:
        """Load and split the dataset into train and validation sets."""
        # Load dataset using Polars
        self.df = pl.read_csv(self.data_path)

        # Split the data into train and validation sets
        train_df, val_df = train_test_split(self.df, test_size=self.test_size, random_state=42)

        self.train_dataset = PolarsDataset(pl.DataFrame(train_df), output_column=self.output_column)
        self.val_dataset = PolarsDataset(pl.DataFrame(val_df), output_column=self.output_column)

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    # Path to dataset
    data_path = "path_to_your_dataset.csv"
    output_column = "label"  # Column name that contains the labels

    # Instantiate and use the DataModule
    datamodule = PolarsDataModule(data_path, output_column, batch_size=32)
    datamodule.setup()

    for batch in datamodule.train_dataloader():
        features, labels = batch
        print(f"Features: {features}, Labels: {labels}")
