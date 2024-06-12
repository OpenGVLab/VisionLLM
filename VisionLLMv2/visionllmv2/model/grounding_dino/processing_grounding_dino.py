# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Grounding DINO.
"""

from typing import List, Optional, Tuple, Union

from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType, is_torch_available


if is_torch_available():
    import torch


def get_phrases_from_posmap(posmaps, input_ids):
    """Get token ids of phrases from posmaps and input_ids.
    Args:
        posmaps (`torch.BoolTensor` of shape `(num_boxes, hidden_size)`):
            A boolean tensor of text-thresholded logits related to the detected bounding boxes.
        input_ids (`torch.LongTensor`) of shape `(sequence_length, )`):
            A tensor of token ids.
    """
    left_idx = 0
    right_idx = 255

    posmaps[:, 0 : left_idx + 1] = False
    posmaps[:, right_idx:] = False

    token_ids = []
    for posmap in posmaps:
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids.append([input_ids[i] for i in non_zero_idx])

    return token_ids


class GroundingDinoProcessor(ProcessorMixin):
    r"""
    Constructs a Grounding DINO processor which wraps a Deformable DETR image processor and a BERT tokenizer into a
    single processor.
    [`GroundingDinoProcessor`] offers all the functionalities of [`GroundingDinoImageProcessor`] and
    [`AutoTokenizer`]. See the docstring of [`~GroundingDinoProcessor.__call__`] and [`~GroundingDinoProcessor.decode`]
    for more information.
    Args:
        image_processor (`GroundingDinoImageProcessor`):
            An instance of [`GroundingDinoImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "GroundingDinoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = True,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`GroundingDinoImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.
        Please refer to the docstring of the above two methods for more information.
        """
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        # Get only text
        if images is None:
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
            return text_encoding

        # add pixel_values
        encoding_image_processor = self.image_processor(images, return_tensors=return_tensors)

        if text is not None:
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
        else:
            text_encoding = None

        if text_encoding is not None:
            # Keeping same order of model_input_names when both images and text
            text_encoding.update(encoding_image_processor)
            encoding_image_processor = text_encoding

        return encoding_image_processor

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.batch_decode with BertTokenizerFast->PreTrainedTokenizer
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.decode with BertTokenizerFast->PreTrainedTokenizer
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.blip.processing_blip.BlipProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def post_process_grounded_object_detection(
        self,
        outputs,
        input_ids,
        box_threshold: float,
        text_threshold: float,
        target_sizes: Union[TensorType, List[Tuple]] = None,
    ):
        """
        Post-process the output of the model to get the grounded object detection results.
        """
        results = self.image_processor.post_process_object_detection(outputs, box_threshold, target_sizes)

        probs = torch.sigmoid(outputs.logits)  # (batch_size, num_queries, 256)

        for idx, (result, prob) in enumerate(zip(results, probs)):
            labels = result["labels"]
            # Assuming that selected bboxes are sorted by confidence due to Hungarian matching loss in training
            prob = prob[: len(labels)]  # len(labels) , 256
            token_ids = get_phrases_from_posmap(prob > text_threshold, input_ids[idx])
            # overrides result labels key
            result["labels"] = self.batch_decode(token_ids)

        return results



