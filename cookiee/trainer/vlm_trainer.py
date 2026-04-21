import torch
from torch.utils.data import Sampler
from transformers.trainer import has_length

from typing import List, Optional

from .base_trainer import BaseTrainer
from ..callbacks import SaveProcessorCallback
from helper import get_logger

logger = get_logger("vlm-trainer")


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        # megabatches.append(sorted(additional_batch))
        padding_needed = megabatch_size - len(last_mm)

        for mb in megabatches:
            if len(mb) > 0 and lengths[mb[0]] > 0:
                first_mm_batch = mb
                break
        repeats = (padding_needed // len(first_mm_batch)) + 1 # 循环取以防不足
        padding_samples = (first_mm_batch * repeats)[:padding_needed]

        additional_batch = last_mm + padding_samples
        assert len(additional_batch) == megabatch_size
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class VLMTrainer(BaseTrainer):
    def __init__(
        self,
        config,
        model,
        processor=None,
        image_processor=None,
        *args, **kwargs
    ):
        super().__init__(config, model, *args, **kwargs)

        if processor is not None and image_processor is not None:
            self.add_callback(SaveProcessorCallback(processor, image_processor))
        else:
            logger.warning("Processor or image processor is not provided, VLM trainer will not save them")
    

    def _get_train_sampler(self, train_dataset: Optional[torch.utils.data.Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        if getattr(self.config, "group_by_modality_length", False):
            #lengths = self.train_dataset.modality_lengths
            modality_lengths = self.train_dataset["modality_lengths"]
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=modality_lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler(train_dataset)

    
    def get_batch_samples(self, epoch_iterator, num_batches: int, device: torch.device):
        if self.config.task != "preference_pretrain":
            return super().get_batch_samples(epoch_iterator, num_batches, device)
        else:
            batch_samples = []
            num_items_in_batch = None

            for _ in range(num_batches):
                try:
                    batch_samples.append(next(epoch_iterator))
                except StopIteration:
                    break

            count_num_items_in_batch = (
                len(batch_samples) > 0
                and "labels" in batch_samples[0]
                and (
                    # num_items_in_batch is passed to model forward
                    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3757
                    self.model_accepts_loss_kwargs
                    # num_items_in_batch is passed to compute_loss_func
                    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3773
                    or self.compute_loss_func is not None
                    # num_items_in_batch is also verified if (self.model_accepts_loss_kwargs or self.compute_loss_func)
                    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3790
                )
            )

            if count_num_items_in_batch:
                # For now we don't support object detection
                try:
                    # just change here to support pairwise dataset
                    num_items_in_batch = sum([(batch["labels"][:len(batch["labels"])//2].ne(-100)).sum() for batch in batch_samples])
                except (TypeError, AttributeError):
                    pass

            if num_items_in_batch is not None:
                if self.args.average_tokens_across_devices:
                    num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()

                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(device)

                    if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                        # In the DataParallel case, convert the scalar tensor into a 1-dim tensor
                        num_items_in_batch = num_items_in_batch.unsqueeze(0)
                    # Divide by number of devices with the same batch
                    if pc := self.accelerator.parallelism_config:
                        num_items_in_batch = num_items_in_batch // pc.non_data_parallel_size

            return batch_samples, num_items_in_batch