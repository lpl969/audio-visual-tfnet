import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from dataloader.data_process import concat_audio, compute_sh_features


class AudioDataset(Dataset):
    def __init__(
        self,
        annotation_path,
        gt_cls_path,
        gt_position_path,
        audio_path,
        image_path,
        image_size_224=224,
        audio_length=9600,
        sh_order=4,
        return_dict=True,
    ):
        super().__init__()
        annotation_path = self._resolve_annotation_path(annotation_path, audio_path)
        with open(annotation_path, "r", encoding="utf-8") as f:
            self.annotation_lines = [line.strip().replace(chr(92), "/") for line in f if line.strip()]
        if not self.annotation_lines:
            raise ValueError("Manifest must not be empty")
        for name in self.annotation_lines:
            if Path(name).is_absolute() or ".." in Path(name).parts or ":" in name:
                raise ValueError("Manifest entries must be relative paths inside the dataset")

        self.gt_cls_path = str(gt_cls_path)
        self.gt_position_path = str(gt_position_path)
        self.audio_path = str(audio_path)
        self.image_path = str(image_path)
        self.image_size_224 = int(image_size_224)
        self.audio_length = int(audio_length)
        self.sh_order = int(sh_order)
        self.return_dict = bool(return_dict)

        class_ids = {int(Path(line).parts[0]) for line in self.annotation_lines}
        self.num_classes = max(class_ids) + 1 if class_ids else 0

    def __len__(self):
        return len(self.annotation_lines)

    @staticmethod
    def _resolve_annotation_path(annotation_path, audio_path):
        annot_p = Path(annotation_path)
        if annot_p.exists():
            return annot_p
        if "fold" in str(annot_p):
            base_dir = Path(audio_path).parent
            candidate = base_dir / "annotation" / "V2" / annot_p.name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    def _resolve_relative_npy(self, base_path: str, file_name: str) -> str:
        return os.path.join(base_path, file_name)

    def _resolve_image_path(self, file_name: str) -> str:
        key = Path(file_name)
        cls_name = key.parts[0]
        stem = key.stem
        return os.path.join(self.image_path, cls_name, f"{stem}.npy")

    @staticmethod
    def _ensure_chw(img_np: np.ndarray) -> np.ndarray:
        if img_np.ndim == 2:
            img_np = img_np[None, ...]
        elif img_np.ndim == 3 and img_np.shape[-1] in (1, 3):
            img_np = np.transpose(img_np, (2, 0, 1))
        if img_np.shape[0] == 1:
            img_np = np.repeat(img_np, 3, axis=0)
        return img_np.astype(np.float32)

    def _load_image(self, img_path: str, target_size: int) -> torch.Tensor:
        img_np = np.load(img_path)
        img_np = self._ensure_chw(img_np)
        img_tensor = torch.from_numpy(img_np)
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0
        if img_tensor.shape[1:] != (target_size, target_size):
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return img_tensor.float()

    def _fix_audio_length(self, audio_np: np.ndarray) -> np.ndarray:
        t = audio_np.shape[-1]
        if t > self.audio_length:
            return audio_np[:, :self.audio_length]
        if t < self.audio_length:
            return np.pad(audio_np, ((0, 0), (0, self.audio_length - t)))
        return audio_np

    def _load_position_and_mask(self, file_name: str):
        pos_np = np.load(self._resolve_relative_npy(self.gt_position_path, file_name)).reshape(-1)
        if pos_np.shape[0] < 3 or not np.isfinite(pos_np[:3]).all():
            return torch.zeros(3, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)
        return torch.tensor(pos_np[:3], dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)

    def __getitem__(self, index):
        file_name = self.annotation_lines[index]
        audio_np = concat_audio(self.audio_path, file_name)
        audio_np = self._fix_audio_length(audio_np)
        shc_np = compute_sh_features(audio_np, order=self.sh_order)

        audio_tensor = torch.from_numpy(audio_np.astype(np.float32))
        shc_tensor = torch.from_numpy(shc_np.astype(np.float32))

        img_path = self._resolve_image_path(file_name)
        image_224 = self._load_image(img_path, self.image_size_224)

        gt_cls = torch.tensor(
            int(np.load(self._resolve_relative_npy(self.gt_cls_path, file_name)).reshape(-1)[0]),
            dtype=torch.long,
        )
        position, position_mask = self._load_position_and_mask(file_name)

        sample = {
            "id": file_name,
            "audio": audio_tensor,
            "shc": shc_tensor,
            "image_224": image_224,
            "label": gt_cls,
            "position": position,
            "position_mask": position_mask,
        }

        if self.return_dict:
            return sample
        return (
            sample["audio"],
            sample["shc"],
            sample["image_224"],
            sample["label"],
            sample["position"],
            sample["position_mask"],
        )
