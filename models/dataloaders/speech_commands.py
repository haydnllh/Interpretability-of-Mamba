import os
import io
import tarfile
import zipfile
import random
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import soundfile as sf

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import requests

SPEECH_COMMANDS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
SPEECH_COMMANDS_SHA256 = "3b93429b20b9f370dfd9a1441a249dbf3e6f6b6f2fa14fd3e0f2d8a39d1f1b02"  # best-effort; not strictly enforced here

DEFAULT_KEYWORDS = ["yes","no","up","down","left","right","on","off","stop","go"]
BACKGROUND_DIRNAME = "_background_noise_"


def _download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {url}") as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_tar_gz(tar_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=out_dir)


def _load_list_file(list_path: Path) -> List[str]:
    if not list_path.exists():
        return []
    with open(list_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


def _gather_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.wav") if BACKGROUND_DIRNAME not in p.parts]


def _read_background_noise(root: Path, sample_rate: int) -> List[torch.Tensor]:
    bg_dir = root / BACKGROUND_DIRNAME
    noises = []
    if bg_dir.exists():
        for p in bg_dir.glob("*.wav"):
            wav, sr = sf.read(p)
            wav = torch.from_numpy(wav).float().unsqueeze(0)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            noises.append(wav)
    return noises


class AudioAugment:
    def __init__(
        self,
        sample_rate: int = 16000,
        max_time_shift_ms: float = 100.0,
        bg_noise_prob: float = 0.8,
        bg_noise_snr_db_range: Tuple[float, float] = (5.0, 20.0),
        volume_jitter_db: float = 3.0,
        background_noises: Optional[List[torch.Tensor]] = None,
    ):
        self.sample_rate = sample_rate
        self.max_shift = int(max_time_shift_ms / 1000.0 * sample_rate)
        self.bg_noise_prob = bg_noise_prob
        self.snr_range = bg_noise_snr_db_range
        self.vol_jitter_db = volume_jitter_db
        self.background_noises = background_noises or []

    @staticmethod
    def _db_to_linear(db: float) -> float:
        return 10 ** (db / 20.0)

    def _time_shift(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0:
            return x
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        if shift > 0:
            return torch.nn.functional.pad(x, (shift, 0))[:, :-shift]
        else:
            shift = -shift
            return torch.nn.functional.pad(x, (0, shift))[:, shift:]

    def _mix_bg_noise(self, x: torch.Tensor) -> torch.Tensor:
        if not self.background_noises or random.random() > self.bg_noise_prob:
            return x
        noise = random.choice(self.background_noises)
        # choose random slice of noise matching x
        if noise.size(1) >= x.size(1):
            start = random.randint(0, noise.size(1) - x.size(1))
            noise_slice = noise[:, start:start + x.size(1)]
        else:
            # tile if noise shorter
            reps = int(np.ceil(x.size(1) / noise.size(1)))
            noise_slice = noise.repeat(1, reps)[:, :x.size(1)]
        # compute scaling by SNR
        x_power = x.pow(2).mean()
        noise_power = noise_slice.pow(2).mean() + 1e-12
        target_snr_db = random.uniform(*self.snr_range)
        snr_linear = 10 ** (target_snr_db / 10.0)
        noise_scale = torch.sqrt(x_power / (snr_linear * noise_power))
        return x + noise_scale * noise_slice

    def _volume_jitter(self, x: torch.Tensor) -> torch.Tensor:
        if self.vol_jitter_db <= 0:
            return x
        gain_db = random.uniform(-self.vol_jitter_db, self.vol_jitter_db)
        return x * self._db_to_linear(gain_db)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self._time_shift(x)
        x = self._mix_bg_noise(x)
        x = self._volume_jitter(x)
        return x


def _pad_or_trim(x: torch.Tensor, num_samples: int) -> torch.Tensor:
    cur = x.size(1)
    if cur == num_samples:
        return x
    if cur > num_samples:
        return x[:, :num_samples]
    # pad
    pad = num_samples - cur
    return torch.nn.functional.pad(x, (0, pad))


class SpeechCommandsDataset(Dataset):
    """
    Standalone PyTorch dataset for Google Speech Commands v2.
    Supports:
    - keywords + unknown + silence classes
    - deterministic splits
    - augmentations with background noise and time-shift
    """
    def __init__(
        self,
        data_dir: str,
        split: str,
        sample_rate: int = 16000,
        duration: float = 1.0,
        target_keywords: Optional[List[str]] = None,
        include_unknown: bool = True,
        include_silence: bool = True,
        augment: bool = False,
        seed: int = 42,
    ):
        assert split in ("train", "val", "test"), "split must be 'train', 'val', or 'test'"
        self.root = Path(data_dir)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.keywords = target_keywords or DEFAULT_KEYWORDS
        self.include_unknown = include_unknown
        self.include_silence = include_silence
        self.augment_enabled = augment and split == "train"
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Ensure dataset exists
        self._prepare_data()

        # Background noises for augmentation and silence samples
        self.background_noises = _read_background_noise(self.root, sample_rate=self.sample_rate)
        self.augment = AudioAugment(sample_rate=self.sample_rate, background_noises=self.background_noises)

        # Build label mapping
        self.label_names = list(self.keywords)
        if self.include_unknown:
            self.label_names.append("unknown")
        if self.include_silence:
            self.label_names.append("silence")
        self.label_to_index: Dict[str, int] = {lbl: i for i, lbl in enumerate(self.label_names)}

        # Load official split lists if present
        val_list = _load_list_file(self.root / "validation_list.txt")
        test_list = _load_list_file(self.root / "testing_list.txt")

        # Build file index
        all_files = _gather_files(self.root)

        # Assign files to splits
        val_set = set(val_list)
        test_set = set(test_list)

        def rel_path(p: Path) -> str:
            return "/".join(p.relative_to(self.root).parts)

        val_files = [p for p in all_files if rel_path(p) in val_set]
        test_files = [p for p in all_files if rel_path(p) in test_set]
        train_files = [p for p in all_files if (rel_path(p) not in val_set and rel_path(p) not in test_set)]

        # If lists missing, make deterministic split
        if len(val_files) == 0 or len(test_files) == 0:
            # group by label directory
            by_label: Dict[str, List[Path]] = {}
            for p in all_files:
                label = p.parent.name
                by_label.setdefault(label, []).append(p)
            train_files, val_files, test_files = [], [], []
            for label, files in by_label.items():
                files = sorted(files, key=lambda x: x.name)
                rng = random.Random(self.seed + hash(label) % (1 << 16))
                rng.shuffle(files)
                n = len(files)
                n_val = max(1, int(0.1 * n))
                n_test = max(1, int(0.1 * n))
                val_files.extend(files[:n_val])
                test_files.extend(files[n_val:n_val + n_test])
                train_files.extend(files[n_val + n_test:])

        # Filter to target classes
        def file_label(p: Path) -> str:
            return p.parent.name

        def is_keyword(lbl: str) -> bool:
            return lbl in self.keywords

        selected: List[Tuple[Path, int]] = []
        if split == "train":
            files = train_files
        elif split == "val":
            files = val_files
        else:
            files = test_files

        for p in files:
            lbl = file_label(p)
            if is_keyword(lbl):
                selected.append((p, self.label_to_index[lbl]))
            else:
                if self.include_unknown and lbl != BACKGROUND_DIRNAME:
                    selected.append((p, self.label_to_index["unknown"]))
                # else drop non-keyword when unknown excluded

        # Add synthetic silence samples from background noise for train/val/test
        self.silence_indices: List[int] = []
        if self.include_silence and len(self.background_noises) > 0:
            # Generate N_silence proportional to dataset size
            n_silence = max(100, int(0.02 * len(selected)))
            # We store sentinel entries with None path and label 'silence'
            for _ in range(n_silence):
                self.silence_indices.append(len(selected))
                selected.append((None, self.label_to_index["silence"]))

        self.items = selected

    def _prepare_data(self):
        self.root.mkdir(parents=True, exist_ok=True)
        archive_path = self.root / "speech_commands_v0.02.tar.gz"
        # Download if missing
        _download_file(SPEECH_COMMANDS_URL, archive_path)
        # Optional: verify checksum; skip strict enforcement to avoid breakage on provider changes
        try:
            sha = _sha256_file(archive_path)
            # print(f"Archive sha256: {sha}")
        except Exception:
            pass
        # Extract if not extracted
        marker = self.root / ".extracted_v0.02"
        if not marker.exists():
            _extract_tar_gz(archive_path, self.root)
            marker.write_text("ok")

    def __len__(self):
        return len(self.items)

    def _load_wave(self, path: Optional[Path]) -> torch.Tensor:
        if path is None:
            # create silence sample from background noise slice with very low amplitude
            if len(self.background_noises) > 0:
                noise = random.choice(self.background_noises)
                # pick a random slice and attenuate strongly
                if noise.size(1) >= self.num_samples:
                    start = random.randint(0, noise.size(1) - self.num_samples)
                    x = noise[:, start:start + self.num_samples]
                else:
                    reps = int(np.ceil(self.num_samples / noise.size(1)))
                    x = noise.repeat(1, reps)[:, :self.num_samples]
                x = x * 0.0  # pure silence; alternatively keep very low amplitude
            else:
                x = torch.zeros(1, self.num_samples)
            return x
        wav, sr = sf.read(path)
        wav = torch.from_numpy(wav).float().unsqueeze(0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = _pad_or_trim(wav, self.num_samples)
        return wav

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        x = self._load_wave(path)
        if self.augment_enabled:
            x = self.augment(x)
        # Return waveform; downstream can transform to spectrogram or MFCC
        return x, label


def collate_fn(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B, 1, T]
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def class_weights_from_counts(counts: List[int]) -> torch.Tensor:
    total = sum(counts)
    weights = [total / (c if c > 0 else 1) for c in counts]
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum() * len(weights)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    sample_rate: int = 16000,
    duration: float = 1.0,
    target_keywords: Optional[List[str]] = None,
    include_unknown: bool = False,
    include_silence: bool = False,
    augment: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_ds = SpeechCommandsDataset(
        data_dir=data_dir,
        split="train",
        sample_rate=sample_rate,
        duration=duration,
        target_keywords=target_keywords,
        include_unknown=include_unknown,
        include_silence=include_silence,
        augment=augment,
        seed=seed,
    )
    val_ds = SpeechCommandsDataset(
        data_dir=data_dir,
        split="val",
        sample_rate=sample_rate,
        duration=duration,
        target_keywords=target_keywords,
        include_unknown=include_unknown,
        include_silence=include_silence,
        augment=False,
        seed=seed,
    )
    test_ds = SpeechCommandsDataset(
        data_dir=data_dir,
        split="test",
        sample_rate=sample_rate,
        duration=duration,
        target_keywords=target_keywords,
        include_unknown=include_unknown,
        include_silence=include_silence,
        augment=False,
        seed=seed,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_ds.label_names