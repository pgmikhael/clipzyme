from typing import Dict
from nox.utils.registry import register_object
from nox.utils.classes import Nox
import numpy as np
import torch
from torch import distributed as dist
import math


@register_object("ranking_metrics", "metric")
class RankingMetrics(Nox):
    def __init__(self, args) -> None:
        super().__init__()

    @property
    def metric_keys(self):
        return []

    def __call__(self, logging_dict, model, args) -> Dict:
        stats_dict = dict()
        rankings = logging_dict["rankings"][0] # TODO: Understand why this is len 1
        ranking = torch.cat(rankings)
        ########### for debugging purposes ##########
        rank = 0
        world_size = 1
        device = logging_dict['num_negatives'][0][0].device 
        #############################################
        num_negatives = logging_dict["num_negatives"][0] # TODO: Understand why this is len 1

        num_negative = torch.cat(num_negatives)
        all_size = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)
        if world_size > 1:  # TODO
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        cum_size = all_size.cumsum(0)
        all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_ranking[cum_size[rank] - all_size[rank] : cum_size[rank]] = ranking
        all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_num_negative[
            cum_size[rank] - all_size[rank] : cum_size[rank]
        ] = num_negative

        if world_size > 1:  # TODO
            dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

        if rank == 0:  # TODO
            stats_dict["mean_rank"] = all_ranking.float().mean()
            stats_dict["mean reciprocal rank"] = (1 / all_ranking.float()).mean()

            for hit_n in args.hits_at_n:
                stats_dict[hit_n] = self.calc_hits_at(
                    all_ranking, all_num_negative, hit_n
                )

        return stats_dict

    def calc_hits_at(self, all_ranking, all_num_negative, hit_n):
        values = hit_n[5:].split("_")
        threshold = int(values[0])
        if len(values) > 1:
            num_sample = int(values[1])
            # unbiased estimation
            fp_rate = (all_ranking - 1).float() / all_num_negative
            score = 0
            for i in range(threshold):
                # choose i false positive from num_sample - 1 negatives
                num_comb = (
                    math.factorial(num_sample - 1)
                    / math.factorial(i)
                    / math.factorial(num_sample - i - 1)
                )
                score += (
                    num_comb * (fp_rate**i) * ((1 - fp_rate) ** (num_sample - i - 1))
                )
            return score.mean()
        else:
            return (all_ranking <= threshold).float().mean()

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--hits_at_n",
            type=str,
            nargs="*",
            default=["hits@1", "hits@3", "hits@10", "hits@10_50"],
            help="Whether to log metrics per class or just log average across classes",
        )
