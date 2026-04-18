import numpy as np
import cv2
import torch
import timm
from torchvision import transforms
from PIL import Image


class FeatureExtractor:
    def __init__(self, device=None):

        print("Loading feature extractor (prov-gigapath)...")

        # --- Device ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        import os

        local_appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        cache_dir = os.path.join(local_appdata, "OffSightPathology", "gigapath_cache")

        # --- Model ---
        # Executing the first time creates a local cache of the model
        self.model = timm.create_model(
            model_name='hf_hub:prov-gigapath/prov-gigapath',
            pretrained=True,
            cache_dir=cache_dir
        ).to(self.device).eval()

        self.num_features = self.model.num_features

        # --- Preprocessing pipeline ---
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # converts to [0,1] and CHW
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        print("Finished loading feature extractor")

    def _grid_sampling(
            self,
            batch: np.ndarray,
            patch_size=256,
    ):

        if batch.ndim != 4 or batch.shape[-1] != 3:
            raise ValueError(f"Expected batch shape (N, H, W, 3), got {batch.shape}")

        all_patches = []
        tile_ids = []

        N = batch.shape[0]

        for i in range(N):
            img = batch[i]

            for _i in range(img.shape[0] // patch_size):
                for _j in range(img.shape[1] // patch_size):
                    patch = img[
                            (_i * patch_size):((_i * patch_size) + patch_size),
                            (_j * patch_size):((_j * patch_size) + patch_size),
                            :
                            ]
                    all_patches.append(patch)

                    tile_ids.append(i)

        patches = np.stack(all_patches, axis=0)
        tile_ids = np.array(tile_ids, dtype=np.int64)
        return patches, tile_ids

    def _preprocess_batch(self, batch: np.ndarray) -> torch.Tensor:
        """
        batch: numpy array of shape (N, H, W, 3) from cv2.imread,
               i.e. BGR uint8 images.
        """
        # Ensure shape is (N, H, W, 3)
        if batch.ndim != 4 or batch.shape[-1] != 3:
            raise ValueError(f"Expected batch shape (N, H, W, 3), got {batch.shape}")

        tensors = []
        for img_bgr in batch:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensors.append(self.preprocess(pil_img))

        return torch.stack(tensors, dim=0)

    def process(self, batch: np.ndarray, return_numpy=True, patch_size=256):
        """
        batch: numpy array from cv2 (N, H, W, 3)
        returns: (N, D) feature matrix (numpy or torch tensor)
        """
        if batch is None or len(batch) == 0:
            # Return empty properly-shaped container if you want,
            # but empty list is also fine.
            return np.empty((0, 0), dtype=np.float32) if return_numpy else torch.empty(0, 0)

        patches, tile_ids = self._grid_sampling(
            batch,
            patch_size=patch_size,
        )

        x = self._preprocess_batch(patches)
        x = x.to(self.device)

        if self.device.type == "cuda":
            # NOTE: amp autocast reduces runtime by 50%+ with minimal downside
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                patch_feats = self.model(x)
        else:
            with torch.inference_mode():
                patch_feats = self.model(x)

        patch_feats = patch_feats.cpu()

        # Aggregate per original tile (mean pooling)
        M, D = patch_feats.shape
        N = batch.shape[0]

        tile_feats = torch.zeros((N, D), dtype=patch_feats.dtype)
        counts = torch.zeros(N, dtype=torch.long)

        for idx in range(M):
            t = tile_ids[idx]
            tile_feats[t] += patch_feats[idx]
            counts[t] += 1

        # avoid division by zero just in case
        counts = counts.clamp(min=1)
        tile_feats = tile_feats / counts.unsqueeze(1)

        if return_numpy:
            return tile_feats.numpy()
        else:
            return tile_feats
