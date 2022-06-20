
import os
import torch

from torch.utils.data import DataLoader

from nff.utils import constants as const
from nff.data import Dataset, collate_dicts
from nff.train import load_model, evaluate


def make_endpoint_dset(info,
                       end_key,
                       cutoff,
                       test=False):

    props = {"key": [], 'nxyz': []}
    for key, sub_dic in info.items():
        summary = sub_dic.get('summary', {}).get(end_key)
        if not summary:
            continue
        props['nxyz'].append(summary['endpoint_nxyz'])
        props['key'].append(key)

    dset = Dataset(props,
                   do_copy=False)
    dset.generate_neighbor_list(cutoff=cutoff,
                                undirected=False)

    if test:
        dset.change_idx(list(range(100)))

    return dset


def null_loss(x, y):
    return torch.Tensor([0])


def eval_dann(dset,
              batch_size,
              model,
              device):

    loader = DataLoader(dset,
                        batch_size=batch_size,
                        collate_fn=collate_dicts)
    results, batches, _ = evaluate(model=model,
                                   loader=loader,
                                   loss_fn=null_loss,
                                   device=device)

    gaps = (torch.cat(results['energy_1']).reshape(-1) -
            torch.cat(results['energy_0']).reshape(-1)
            ).numpy() / const.EV_TO_KCAL_MOL

    keys = batches['key']
    assert len(keys) == len(gaps)
    gap_dic = {key: float(gap) for key, gap in zip(keys, gaps)}

    return gap_dic
