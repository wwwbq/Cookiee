import torch

from collections import defaultdict
from typing import Any, Dict, Iterable

from .base import LossOutput


class MetricAggregator:
    def _to_number(self, value):
        if torch.is_tensor(value):
            return value.detach().item()
        return value

    def merge_scalar_metrics(self, gathered_objects: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        scalar_metrics = defaultdict(list)

        for obj in gathered_objects:
            for name, value in (obj.get("scalar_metrics") or {}).items():
                scalar_metrics[name].append(float(self._to_number(value)))

        merged_metrics = {}
        for name, values in scalar_metrics.items():
            if len(values) > 0:
                merged_metrics[name] = sum(values) / len(values)

        return merged_metrics

    def merge_grouped_metrics(self, gathered_objects: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        grouped_metrics = defaultdict(dict)

        for obj in gathered_objects:
            for group_name, group in (obj.get("grouped_metrics") or {}).items():
                grouped_metrics[group_name] = grouped_metrics[group_name] or {}
                for metric_name, stats in group.items():
                    grouped_metrics[group_name].setdefault(metric_name, {})
                    for stat_name, stat_value in stats.items():
                        stat_value = self._to_number(stat_value)
                        if stat_name in {"sum", "count"}:
                            grouped_metrics[group_name][metric_name][stat_name] = (
                                grouped_metrics[group_name][metric_name].get(stat_name, 0.0) + stat_value
                            )
                        else:
                            grouped_metrics[group_name][metric_name][stat_name] = stat_value

        merged_metrics = {}

        # grouped metric 主要服务类似 channel loss 这种需要先聚合 sum/count 再求平均的场景。
        for group in grouped_metrics.values():
            for metric_name, stats in group.items():
                if "sum" in stats and "count" in stats:
                    merged_metrics[metric_name] = stats["sum"] / max(stats["count"], 1e-12)
                else:
                    for stat_name, stat_value in stats.items():
                        merged_metrics[f"{metric_name}.{stat_name}"] = stat_value

        return merged_metrics

    def merge(self, gathered_objects: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        merged_metrics = {}
        merged_metrics.update(self.merge_scalar_metrics(gathered_objects))
        merged_metrics.update(self.merge_grouped_metrics(gathered_objects))
        return merged_metrics

    def scale_metrics(self, metrics: Dict[str, float], loss_func, accelerator) -> Dict[str, float]:
        scaled_metrics = dict(metrics)
        for key, need_scale in getattr(loss_func, "scaled_metrics", {}).items():
            if need_scale and key in scaled_metrics:
                scaled_metrics[key] = scaled_metrics[key] * accelerator.num_processes
        return scaled_metrics

    def has_scalar_metrics(self, gathered_objects: Iterable[Dict[str, Any]]) -> bool:
        return any(obj.get("scalar_metrics") is not None for obj in gathered_objects)

    def has_grouped_metrics(self, gathered_objects: Iterable[Dict[str, Any]]) -> bool:
        return any(obj.get("grouped_metrics") is not None for obj in gathered_objects)

    def __call__(self, accelerator, output: LossOutput) -> Dict[str, float]:
        gathered_objects = accelerator.gather_for_metrics(
            output.to_gather_object(),
            use_gather_object=True,
        )
        merged_metrics = {}

        if self.has_scalar_metrics(gathered_objects):
            merged_metrics.update(self.merge_scalar_metrics(gathered_objects))

        if self.has_grouped_metrics(gathered_objects):
            merged_metrics.update(self.merge_grouped_metrics(gathered_objects))

        return merged_metrics
