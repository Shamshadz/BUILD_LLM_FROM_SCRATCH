import torch
import tiktoken
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    """Create input and next-token target windows from text."""

    def __init__(self, txt, tokenizer, max_length, stride):
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")
        if stride <= 0:
            raise ValueError("stride must be greater than zero")

        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=120,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    """Create GPT-2 token windows and return a PyTorch DataLoader."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
