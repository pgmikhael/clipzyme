import torch
import pickle
from p_tqdm import p_map
from functools import partial


def collect_screening_results(config):
    def read_protein(u, path):
        try:
            d = pickle.load(
                open(
                    f"/data/rsg/mammogram/pgmikhael/clip_hiddens/{path}/sample_{u}.hiddens",
                    "rb",
                )
            )
            return d["hidden"]
        except:
            return

    screening_set = pickle.load(
        open(
            "/Mounts/rbg-storage1/datasets/Enzymes/uniprot2sequence_standard_set_structs.p",
            "rb",
        )
    )
    screening_set = {k: v for k, v in screening_set.items() if v != ""}
    # has structure and msa
    alphafold_files = pickle.load(
        open("/Mounts/rbg-storage1/datasets/Metabo/alphafold_enzymes.p", "rb")
    )
    msa_files = pickle.load(
        open("/Mounts/rbg-storage1/datasets/Enzymes/uniprot2msa_embedding.p", "rb")
    )
    screening_set = {
        k: v
        for k, v in screening_set.items()
        if (k in alphafold_files) and (k in msa_files) and (len(v) <= 650)
    }
    len(screening_set)
    screening_set_uniprots = list(screening_set.keys())

    experiment_name = "d8faada03e032e26e2216ea83a88529cepoch=22"

    read_protein_func = partial(read_protein, path=experiment_name)
    hiddens = p_map(read_protein_func, screening_set_uniprots)

    all_ec_uniprots_ = [
        u for u, h in zip(screening_set_uniprots, hiddens) if h is not None
    ]
    hiddens = [h for h in hiddens if h is not None]
    hiddens = torch.stack(hiddens)

    pickle.dump(
        {"hiddens": hiddens, "uniprots": all_ec_uniprots_},
        open(
            f"/data/rsg/mammogram/pgmikhael/notebooks/EScreening/precomputed_{experiment_name}.p",
            "wb",
        ),
    )
