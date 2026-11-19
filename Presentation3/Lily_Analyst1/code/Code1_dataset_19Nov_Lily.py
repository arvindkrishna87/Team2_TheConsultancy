"""
Dataset classes for MIL training

Presentation 2, Analyst choice 1 changes:
  * Every patch tensor travels with its own filename. __getitem__ returns
    "stain_paths", aligned row-for-row with "stain_slices", so attention scores
    can be matched to images by name instead of by sort order.
  * No silent failures: a missing/unreadable embedding, an empty slice, or a
    malformed tensor raises an error that names the file, instead of being skipped.

Presentation 3, Analyst choice 1 changes (training set only; both default to 0 = off):
  * stain_dropout=p: with probability p, one stain (chosen uniformly among the
    stains the case has) is removed before its patches are even loaded. A case
    with a single stain keeps it. The stain comes back as an empty list, the same
    format as a case that genuinely lacks that stain.
  * slice_dropout=p: each slice of each stain is removed with probability p,
    always keeping at least one slice per stain.
  * A new draw is made on every __getitem__ call (cases are not cached), so a case
    loses a different stain/slice in each epoch.
  * "dropped_stain" and "dropped_slices" are returned for logging only.
"""
import os
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple

from config import MODEL_CONFIG


class StainBagCasePooledFeatureDataset(Dataset):
    """
    Loads precomputed pooled feature vectors (.pt) for each patch.

    For each CASE returns:
    {
        "case_id": int/str,
        "stain_slices": {
            "h&e": [Tensor(P1, F4), Tensor(P2, F4), ...],
            ...
        },
        "stain_paths": {                     # NEW: same nesting as stain_slices
            "h&e": [[png_path_row0, png_path_row1, ...], ...],
            ...
        },
        "label": LongTensor scalar
    }
    stain_paths[stain][s][i] is the patch image behind stain_slices[stain][s][i].
    """

    def __init__(
        self,
        case_dict: Dict[Any, Dict[str, List[List[str]]]],
        label_map: Dict[Any, int],
        embeddings_dir: str,
        stains: Tuple[str, ...] = None,
        per_slice_cap: Optional[int] = None,
        max_slices_per_stain: Optional[int] = None,
        shuffle_patches: bool = True,
        feature_dtype: torch.dtype = torch.float32,
        stain_dropout: float = 0.0,
        slice_dropout: float = 0.0,
    ):
        if not 0.0 <= stain_dropout <= 1.0 or not 0.0 <= slice_dropout < 1.0:
            raise ValueError(f"stain_dropout must be in [0, 1] and slice_dropout in [0, 1); "
                             f"got {stain_dropout}, {slice_dropout}")
        self.stain_dropout = stain_dropout
        self.slice_dropout = slice_dropout
        # drop_empty_slices was removed: empty slices now raise (see __getitem__).
        self.embeddings_dir = embeddings_dir
        self.stains = list(stains) if stains else list(MODEL_CONFIG["stains"])
        self.per_slice_cap = per_slice_cap if per_slice_cap is not None else MODEL_CONFIG["per_slice_cap"]
        self.max_slices_per_stain = (
            max_slices_per_stain if max_slices_per_stain is not None else MODEL_CONFIG["max_slices_per_stain"]
        )
        self.shuffle_patches = shuffle_patches
        self.feature_dtype = feature_dtype

        self.items = []
        for case_id, stain_map in case_dict.items():
            if case_id not in label_map:
                raise KeyError(f"Case {case_id} has patches but no label in label_map")
            norm_map = {k.lower(): v for k, v in stain_map.items()}
            self.items.append((case_id, norm_map))
        self.label_map = label_map

    def __len__(self):
        return len(self.items)

    def _patch_path_to_feat_path(self, patch_path: str) -> str:
        # patch_path is a full path like /projects/.../patches/<fname>.png
        fname = os.path.basename(patch_path)
        feat_name = os.path.splitext(fname)[0] + ".pt"
        return os.path.join(self.embeddings_dir, feat_name)

    def _load_patch_vector(self, patch_path: str) -> torch.Tensor:
        fp = self._patch_path_to_feat_path(patch_path)
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Embedding not found for patch {patch_path}: {fp}")
        try:
            v = torch.load(fp, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Could not load embedding {fp} (patch {patch_path})") from e
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"Embedding {fp} is {type(v).__name__}, expected a tensor")
        if v.dim() != 1:
            v = v.view(-1)
        if v.dtype != self.feature_dtype:
            v = v.to(self.feature_dtype)  # fp16 embeddings -> fp32 model
        return v

    def _load_slice_tensor(self, patch_paths: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """Return (tensor of shape (P, F), the P patch paths in the same row order)."""
        patch_paths = list(patch_paths)

        # Patches with no embedding (e.g. smaller than 32x32, skipped when embeddings were made)
        # are left out, and every one is printed so nothing is dropped silently.
        missing = [p for p in patch_paths if not os.path.isfile(self._patch_path_to_feat_path(p))]
        if missing:
            print(f"EXCLUDED {len(missing)} patch(es) with no embedding: "
                  + ", ".join(os.path.basename(m) for m in missing))
            patch_paths = [p for p in patch_paths if p not in missing]
        if not patch_paths:
            raise ValueError("Every patch in this slice is missing its embedding")

        if self.shuffle_patches:
            random.shuffle(patch_paths)  # paths are shuffled, so names follow their rows
        if self.per_slice_cap and len(patch_paths) > self.per_slice_cap:
            patch_paths = patch_paths[: self.per_slice_cap]

        vecs = [self._load_patch_vector(p) for p in patch_paths]

        sizes = {v.numel() for v in vecs}
        if len(sizes) != 1:
            raise ValueError(f"Embeddings in one slice have different lengths {sorted(sizes)}: "
                             f"{os.path.basename(patch_paths[0])} ...")
        return torch.stack(vecs, dim=0), patch_paths

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id, stain_map = self.items[idx]
        label = torch.tensor(self.label_map[case_id], dtype=torch.long)

        # --- Stain dropout: decided before loading, so a dropped stain is never read ---
        present = [s for s in self.stains if stain_map.get(s)]
        dropped_stain = None
        if self.stain_dropout > 0 and len(present) >= 2 and random.random() < self.stain_dropout:
            dropped_stain = random.choice(present)

        stain_slices: Dict[str, List[torch.Tensor]] = {}
        stain_paths: Dict[str, List[List[str]]] = {}
        dropped_slices: Dict[str, List[int]] = {}
        for stain in self.stains:
            if stain == dropped_stain:
                # Same format as a case that has no slices for this stain
                stain_slices[stain] = []
                stain_paths[stain] = []
                continue

            slice_lists = stain_map.get(stain, []) or []

            if self.max_slices_per_stain is not None and len(slice_lists) > self.max_slices_per_stain:
                slice_lists = slice_lists[: self.max_slices_per_stain]

            # --- Slice dropout: drop each slice with prob p, keep at least one ---
            if self.slice_dropout > 0 and len(slice_lists) >= 2:
                keep = [i for i in range(len(slice_lists)) if random.random() >= self.slice_dropout]
                if not keep:
                    keep = [random.randrange(len(slice_lists))]
                dropped_slices[stain] = [i for i in range(len(slice_lists)) if i not in keep]
                slice_lists = [slice_lists[i] for i in keep]

            tensors_for_stain: List[torch.Tensor] = []
            paths_for_stain: List[List[str]] = []
            for s_i, sl in enumerate(slice_lists):
                if not sl:
                    raise ValueError(f"Case {case_id}, stain {stain}, slice {s_i} has no patches")
                sl_tensor, sl_paths = self._load_slice_tensor(sl)
                tensors_for_stain.append(sl_tensor)
                paths_for_stain.append(sl_paths)

            stain_slices[stain] = tensors_for_stain
            stain_paths[stain] = paths_for_stain

        return {
            "case_id": case_id,
            "stain_slices": stain_slices,
            "stain_paths": stain_paths,
            "label": label,
            "dropped_stain": dropped_stain,      # logging only
            "dropped_slices": dropped_slices,    # logging only: original slice indices
        }


def case_collate_fn(batch):
    """
    Collate function for MIL dataset.
    Keeps variable-length structures (including stain_paths) as-is;
    use batch_size=1 to avoid padding/masking.
    """
    return batch