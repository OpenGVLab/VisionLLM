import torch
import transformers
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List

from visionllmv2.constant import IGNORE_INDEX

# For LLava image dataset
@dataclass
class DataCollatorForImageDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        return batch

# For GPT4ROI image/region dataset
@dataclass
class DataCollatorForHybridDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # llava data
        meta_keys = ('input_ids', 'labels', 'image', 'bboxes')
        input_ids, labels, images, bboxes = tuple(
            [instance.get(key, None) for instance in instances] for key in meta_keys)
        if all([x is not None for x in images]):
            images = torch.stack(images)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
            bboxes=bboxes
        )
        return batch



# For Llava/GPT4ROI/COCO image/region/coco dataset
@dataclass
class DataCollatorForHybridDetDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        meta_keys = ('input_ids', 'labels', 'image', 'bboxes')
        input_ids, labels, images, bboxes = tuple(
            [instance.get(key, None) for instance in instances] for key in meta_keys)
        if all([x is not None for x in images]):
            images = torch.stack(images)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
            bboxes=bboxes
        )
        # det data
        if 'image_aug' in instances[0]:
            keys = ('image_aug', 'class_labels', 'boxes', 'img_metas')
            images_aug, class_labels, boxes, img_metas = tuple(
                [instance.get(key, None) for instance in instances] for key in keys
            )
            # images_aug (list[tensor]), targets (list[dict]), img_metas (list[dict])
            images_aug, img_metas = list(images_aug), list(img_metas)
            if class_labels[0] is not None: # train 
                targets = [{'class_labels': label, 'boxes': box} for label, box in zip(class_labels, boxes)]
                batch.update(
                    {
                        'images_aug': images_aug, 'targets': targets, 'img_metas': img_metas
                    }
                )
            else:                           # inference
                batch.update(
                    {
                        'images_aug': images_aug, 'img_metas': img_metas
                    }
                )
        return batch

# For Llava/GPT4ROI/COCO image/region/coco dataset
@dataclass
class DataCollatorForHybridDetSegDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # ====================== chat =========================
        meta_keys = ('input_ids', 'labels', 'image', 'bboxes')
        input_ids, labels, images, bboxes = tuple(
            [instance.get(key, None) for instance in instances] for key in meta_keys)
        if all([x is not None for x in images]):
            images = torch.stack(images)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
            bboxes=bboxes
        )
        # ========================= det/seg ====================================
        if 'image_aug' in instances[0]:
            keys = ('image_aug', 'class_labels', 'boxes', 'mask_labels', 'img_metas')
            images_aug, class_labels, boxes, mask_labels, img_metas = tuple(
                [instance.get(key, None) for instance in instances] for key in keys
            )
            # images_aug (list[tensor]), targets (list[dict]), img_metas (list[dict])
            images_aug, img_metas = list(images_aug), list(img_metas)
            if class_labels[0] is not None: # train 
                if boxes[0] is not None and mask_labels[0] is not None:  # have box and mask annotations
                    targets = [{'class_labels': label, 'boxes': box, 'mask_labels': mask} for label, box, mask in zip(class_labels, boxes, mask_labels)]
                elif boxes[0] is not None:  # have box annotations
                    targets = [{'class_labels': label, 'boxes': box} for label, box in zip(class_labels, boxes)]
                elif mask_labels[0] is not None:  # have mask annotations
                    targets = [{'class_labels': label, 'mask_labels': mask} for label, mask in zip(class_labels, mask_labels)]
                batch.update(
                    {
                        'images_aug': images_aug, 'targets': targets, 'img_metas': img_metas
                    }
                )
            else:                           # inference
                batch.update(
                    {
                        'images_aug': images_aug, 'img_metas': img_metas
                    }
                )
        return batch


# For Llava/GPT4ROI/COCO image/region/coco det/seg/pose dataset
@dataclass
class DataCollatorForHybridDetSegPoseDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # ====================== chat =========================
        meta_keys = ('input_ids', 'labels', 'image')
        input_ids, labels, images = tuple(
            [instance.get(key, None) for instance in instances] for key in meta_keys)
        if all([x is not None for x in images]):
            images = torch.stack(images)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
        )
        # ========================= region ====================================
        if 'regions' in instances[0]:
            keys = ('regions', 'img_metas')
            regions, img_metas = tuple(
                [instance.get(key, None) for instance in instances] for key in keys
            )
            regions, img_metas = list(regions), list(img_metas)  # list[tensor] of [n, h, w], list[dict]
            batch.update(
                    {
                        'regions': regions, 'img_metas': img_metas
                    }
                )
        # ========================= det/seg/pose ====================================
        if 'image_aug' in instances[0]:
            keys = ('image_aug', 'class_labels', 'boxes', 'mask_labels', 'keypoints', 'area', 'img_metas')
            images_aug, class_labels, boxes, mask_labels, keypoints, areas, img_metas = tuple(
                [instance.get(key, None) for instance in instances] for key in keys
            )
            # images_aug (list[tensor]), targets (list[dict]), img_metas (list[dict])
            images_aug, img_metas = list(images_aug), list(img_metas)
            if class_labels[0] is not None: # train 
                if boxes[0] is not None and mask_labels[0] is not None:  # have box and mask annotations
                    targets = [{'class_labels': label, 'boxes': box, 'mask_labels': mask} for label, box, mask in zip(class_labels, boxes, mask_labels)]
                elif boxes[0] is not None and keypoints[0] is not None:  # have box and keypoint annotations
                    targets = [{'class_labels': label, 'boxes': box, 'keypoints': keypoint, 'area': area} for label, box, keypoint, area in zip(class_labels, boxes, keypoints, areas)]
                elif boxes[0] is not None:  # have box annotations
                    targets = [{'class_labels': label, 'boxes': box} for label, box in zip(class_labels, boxes)]
                batch.update(
                    {
                        'images_aug': images_aug, 'targets': targets, 'img_metas': img_metas
                    }
                )
            else:                           # inference
                batch.update(
                    {
                        'images_aug': images_aug, 'img_metas': img_metas
                    }
                )
        return batch
    

# For Llava/GPT4ROI/COCO image/region/coco det/seg/pose dataset, can process 'anyres'
@dataclass
class DataCollatorForHybridDetSegPoseDatasetV2(object):

    tokenizer: transformers.PreTrainedTokenizer
    image_aspect_ratio: str

    def __call__(self, instances):
        # ====================== chat =========================
        meta_keys = ('input_ids', 'labels', 'image')
        input_ids, labels, images = tuple(
            [instance.get(key, None) for instance in instances] for key in meta_keys)
        if self.image_aspect_ratio == 'anyres':
            if all([x is not None for x in images]):
                images = list(images)   # list[tensor], bs x [1 + n_split, 3, h, w]
        else:
            if all([x is not None for x in images]):
                images = torch.stack(images)  # tensor: [bs, 3, h, w]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
        )
        # ========================= region ====================================
        if 'regions' in instances[0]:
            keys = ('regions', 'img_metas')
            regions, img_metas = tuple(
                [instance.get(key, None) for instance in instances] for key in keys
            )
            regions, img_metas = list(regions), list(img_metas)  # list[tensor] of [n, h, w], list[dict]
            batch.update(
                    {
                        'regions': regions, 'img_metas': img_metas
                    }
                )
        # ========================= det/seg/pose ====================================
        if 'image_aug' in instances[0]:
            keys = ('image_aug', 'class_labels', 'boxes', 'mask_labels', 'keypoints', 'area', 'img_metas')
            images_aug, class_labels, boxes, mask_labels, keypoints, areas, img_metas = tuple(
                [instance.get(key, None) for instance in instances] for key in keys
            )
            # images_aug (list[tensor]), targets (list[dict]), img_metas (list[dict])
            images_aug, img_metas = list(images_aug), list(img_metas)
            if class_labels[0] is not None: # train 
                if boxes[0] is not None and mask_labels[0] is not None:  # have box and mask annotations
                    targets = [{'class_labels': label, 'boxes': box, 'mask_labels': mask} for label, box, mask in zip(class_labels, boxes, mask_labels)]
                elif boxes[0] is not None and keypoints[0] is not None:  # have box and keypoint annotations
                    targets = [{'class_labels': label, 'boxes': box, 'keypoints': keypoint, 'area': area} for label, box, keypoint, area in zip(class_labels, boxes, keypoints, areas)]
                elif boxes[0] is not None:  # have box annotations
                    targets = [{'class_labels': label, 'boxes': box} for label, box in zip(class_labels, boxes)]
                batch.update(
                    {
                        'images_aug': images_aug, 'targets': targets, 'img_metas': img_metas
                    }
                )
            else:                           # inference
                batch.update(
                    {
                        'images_aug': images_aug, 'img_metas': img_metas
                    }
                )
        return batch


# For Llava/GPT4ROI/COCO image/region/coco det/seg/pose/gen/edit dataset, can process 'anyres'
@dataclass
class DataCollatorForHybridDetSegPoseGenDataset(object):

    tokenizer: transformers.PreTrainedTokenizer
    image_aspect_ratio: str

    def __call__(self, instances):
        # ====================== chat =========================
        meta_keys = ('input_ids', 'labels', 'image', 'num_splits')
        input_ids, labels, images, num_splits = tuple(
            [instance.get(key, None) for instance in instances] for key in meta_keys)
        if num_splits[0] is not None:   # mmic-data, multiple images in a sample
            images = list(images)  # list[tensor], bs x [n_images_and_splits, 3, h, w]
        else:  # single image in a sample
            if self.image_aspect_ratio == 'anyres':
                if all([x is not None for x in images]):
                    images = list(images)   # list[tensor], bs x [1 + n_split, 3, h, w]
            else:
                if all([x is not None for x in images]):
                    images = torch.stack(images)  # tensor: [bs, 3, h, w]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
        )
        if 'num_splits' in instances[0]: # mmic-data
            batch.update(
                {'num_splits': num_splits}
            )
        # ========================= region ====================================
        if 'regions' in instances[0]:
            keys = ('regions', 'img_metas')
            regions, img_metas = tuple(
                [instance.get(key, None) for instance in instances] for key in keys
            )
            regions, img_metas = list(regions), list(img_metas)  # list[tensor] of [n, h, w], list[dict]
            batch.update(
                    {
                        'regions': regions, 'img_metas': img_metas
                    }
                )
        # ========================= generation ====================================
        if 'output_image' in instances[0]:
            keys = ('input_image', 'output_image', 'caption', 'img_metas')
            input_images, output_images, captions, img_metas = tuple(
                [instance.get(key, None) for instance in instances] for key in keys
            )
            if all([x is not None for x in input_images]):
                input_images = torch.stack(input_images, dim=0) # [bs, 3, h, w]
            output_images = torch.stack(output_images, dim=0)   # [bs, 3, h, w]
            captions = list(captions)   # list[str]
            img_metas = list(img_metas) # list[dict]
            batch.update(
                    {
                        'input_images': input_images, 'output_images': output_images, \
                        'captions': captions, 'img_metas': img_metas
                    }
                )
        # ========================= det/seg/pose ====================================
        if 'image_aug' in instances[0]:
            keys = ('image_aug', 'class_labels', 'boxes', 'mask_labels', 'keypoints', 'area', 'img_metas')
            images_aug, class_labels, boxes, mask_labels, keypoints, areas, img_metas = tuple(
                [instance.get(key, None) for instance in instances] for key in keys
            )
            # images_aug (list[tensor]), targets (list[dict]), img_metas (list[dict])
            images_aug, img_metas = list(images_aug), list(img_metas)
            if class_labels[0] is not None: # train 
                if boxes[0] is not None and mask_labels[0] is not None:  # have box and mask annotations
                    targets = [{'class_labels': label, 'boxes': box, 'mask_labels': mask} for label, box, mask in zip(class_labels, boxes, mask_labels)]
                elif boxes[0] is not None and keypoints[0] is not None:  # have box and keypoint annotations
                    targets = [{'class_labels': label, 'boxes': box, 'keypoints': keypoint, 'area': area} for label, box, keypoint, area in zip(class_labels, boxes, keypoints, areas)]
                elif boxes[0] is not None:  # have box annotations
                    targets = [{'class_labels': label, 'boxes': box} for label, box in zip(class_labels, boxes)]
                batch.update(
                    {
                        'images_aug': images_aug, 'targets': targets, 'img_metas': img_metas
                    }
                )
            else:                           # inference
                batch.update(
                    {
                        'images_aug': images_aug, 'img_metas': img_metas
                    }
                )
        return batch