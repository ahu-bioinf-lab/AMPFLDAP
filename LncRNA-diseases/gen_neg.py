import numpy as np
import scipy.sparse as sp
import os
import torch
import sys

"""预处理脚本"""
def main(prefix):

    pos_pairs_offset = np.load(os.path.join(prefix, "pos_pairs_offsetnew.npz"))
    # pos_pairs_offset = np.load(os.path.join(prefix, "pos_ratings_offset01.npy"))
    neg_ratings_offset = np.load(os.path.join(prefix, "neg_ratings_offset.npy"))
    # neg_ratings_offset02 = np.load(os.path.join(prefix, "neg_ratings_offset02.npy"))
    neg_ratings_offset[:, 1] += 861

    # train_len = pos_pairs_offset['train'].shape[0]
    # val_len = pos_pairs_offset['val'].shape[0]
    # test_len = pos_pairs_offset['test'].shape[0]
    # pos_len = train_len + val_len + test_len
    #
    # indices = np.arange(neg_ratings_offset.shape[0])
    # np.random.shuffle(indices)
    # np.savez(os.path.join(prefix, "neg_pairs_offset"), train=neg_ratings_offset[indices[:train_len]],
    #                                                     val=neg_ratings_offset[indices[train_len:train_len + val_len]],
    #                                                     test=neg_ratings_offset[indices[train_len + val_len:pos_len]])
    train_len = pos_pairs_offset['train'].shape[0]
    val_len = pos_pairs_offset['val'].shape[0]
    indices = np.arange(neg_ratings_offset.shape[0])
    np.random.shuffle(indices)
    # np.savez(os.path.join(prefix, "neg_pairs_offsetnew02"), train=neg_ratings_offset02[indices[:train_len]],val=neg_ratings_offset02[indices[train_len:train_len + val_len]])

    np.savez(os.path.join(prefix, "neg_pairs_offsetnew03"), train=neg_ratings_offset[indices[:train_len]],test =neg_ratings_offset[indices[train_len:train_len + val_len]] ,val=neg_ratings_offset)

    # np.savez(os.path.join(prefix, "neg_pairs_offset01"), train=neg_ratings_offset01[indices[:train_len]],
    #          val=neg_ratings_offset01[indices[train_len:train_len + val_len]],
    #          test=neg_ratings_offset01[indices[train_len + val_len:pos_len]])

if __name__ == '__main__':
    prefix = os.path.join("./preprocessed/LncRNA", 'dataset2')
    main(prefix)
