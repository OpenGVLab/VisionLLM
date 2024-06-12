from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DeepPCBDataset(CocoDataset):
    CLASSES = (
        "open",
        "short",
        "mousebite",
        "spur",
        "copper",
        "pin-hole",
    )

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100)]