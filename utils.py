import numpy as np
import torch
import os
from torchtracer import Tracer


def sort_batch(batch, targets, lengths, ids):
    """
    Sort a minibatch by the length of the sequences with the longest sequences first
    return the sorted batch targes and sequence lengths.
    This way the output can be used by pack_padded_sequences(...)
    """
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    target_tensor = targets[perm_idx]
    sort_ids = ids[perm_idx]
    return seq_tensor, target_tensor, seq_lengths, sort_ids


def pad_and_sort_batch(DataLoaderBatch):
    """
    DataLoaderBatch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest, 
    """
    batch_size = len(DataLoaderBatch)
    batch_split = list(zip(*DataLoaderBatch))

    seqs, targs, lengths, ids = batch_split[0], batch_split[1], batch_split[2], batch_split[3]
    # print(lengths)
    max_length = max(lengths)

    padded_seqs = np.zeros((batch_size, seqs[0].shape[0], max_length), dtype='float32')

    for i, l in enumerate(lengths):
        # print(seqs[i].shape)
        padded_seqs[i, :, 0:l] = seqs[i]

    return sort_batch(torch.tensor(padded_seqs), torch.tensor(targs), torch.tensor(lengths), np.array(ids))


def save_checkpoint(model, optimizer, path, string):
    torch.save(model.state_dict(), os.path.join(path, 'model_{}.pth'.format(string)))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pth'.format(string)))
