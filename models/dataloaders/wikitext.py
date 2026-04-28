import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
import os


class WikiTextDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, block_size: int, stride=None):
        self.data       = token_ids
        self.block_size = block_size
        self.stride = stride or block_size

    def __len__(self):
        return (len(self.data) - self.block_size) // self.stride

    def __getitem__(self, i):
        start = i * self.stride
        chunk = self.data[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]



def _char_tokenize(texts: list[str]):
    full_text = "\n".join(texts)
    chars     = sorted(set(full_text))
    stoi      = {c: i for i, c in enumerate(chars)}
    ids       = torch.tensor([stoi[c] for c in full_text], dtype=torch.long)
    meta      = {
        "vocab_size": len(chars),
        "stoi": stoi,
        "itos": {i: c for c, i in stoi.items()},
        "tokenizer": "char",
    }
    return ids, meta


def _bpe_tokenize(texts: list[str]):
    from transformers import AutoTokenizer
    tok       = AutoTokenizer.from_pretrained("gpt2")
    full_text = "\n".join(texts)
    ids       = tok.encode(full_text)
    ids       = torch.tensor(ids, dtype=torch.long)
    meta      = {
        "vocab_size": tok.vocab_size,
        "tokenizer": "gpt2",
        "hf_tokenizer": tok,      
    }
    return ids, meta


def get_dataloaders(
    dataset_name : str  = "wikitext-103-raw-v1",  
    tokenizer    : str  = "char",              
    block_size   : int  = 256,
    batch_size   : int  = 16,
    num_workers  : int  = 0,
    shuffle_train: bool = True,
):
    name = dataset_name.split("-raw")[0]
    if os.path.exists(f"/scratch/lhl1g23/mamba/data/{name}"):
        raw = load_from_disk(f"/scratch/lhl1g23/mamba/data/{name}")
    else:
        raw = load_dataset("wikitext", dataset_name)
        raw.save_to_disk(f"/scratch/lhl1g23/mamba/data/{name}")

    train_texts = raw["train"]["text"]
    val_texts   = raw["validation"]["text"]

    if tokenizer == "char":
        full_train = "\n".join(train_texts)
        full_val   = "\n".join(val_texts)
        chars      = sorted(set(full_train + full_val))
        stoi       = {c: i for i, c in enumerate(chars)}
        encode     = lambda text: torch.tensor([stoi[c] for c in text], dtype=torch.long)
        train_ids  = encode(full_train)
        val_ids    = encode(full_val)
        meta       = {
            "vocab_size": len(chars),
            "stoi": stoi,
            "itos": {i: c for c, i in stoi.items()},
            "tokenizer": "char",
        }

    elif tokenizer == "gpt2":
        from transformers import AutoTokenizer
        if os.path.exists("/scratch/lhl1g23/mamba/tokenizers/gpt2"):
            tok = AutoTokenizer.from_pretrained(
                "/scratch/lhl1g23/mamba/tokenizers/gpt2",
                local_files_only=True
            )
        else:
            tok = AutoTokenizer.from_pretrained("gpt2")
            tok.save_pretrained("/scratch/lhl1g23/mamba/tokenizers/gpt2")
        
        def encode_in_chunks(texts, chunk_size=1000):
            ids = []
            for i in range(0, len(texts), chunk_size):
                chunk = "\n".join(texts[i:i+chunk_size])
                ids.extend(tok.encode(chunk))
            return torch.tensor(ids, dtype=torch.long)
        
        train_ids = encode_in_chunks(train_texts)
        val_ids   = encode_in_chunks(val_texts)
            
        """ encode     = lambda texts: torch.tensor(
            tok.encode("\n".join(texts)), dtype=torch.long
        )
        train_ids  = encode(train_texts)
        val_ids    = encode(val_texts) """
        meta       = {
            "vocab_size"   : tok.vocab_size,
            "tokenizer"    : "gpt2",
            "hf_tokenizer" : tok,
        }

    else:
        raise ValueError(f"Unknown tokenizer '{tokenizer}'. Choose 'char' or 'gpt2'.")

    train_ds = WikiTextDataset(train_ids, block_size)
    val_ds   = WikiTextDataset(val_ids,   block_size)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = shuffle_train,
        drop_last   = True,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        drop_last   = True,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    return train_loader, val_loader, meta