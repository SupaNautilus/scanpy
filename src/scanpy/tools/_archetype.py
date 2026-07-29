from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix  # noqa: TID251

if TYPE_CHECKING:
    from anndata import AnnData


def archetype(
    adata: AnnData,
    w=None,
    s=None,
):
    """Compute archetype scores and store them in adata.obsm["archetypes"]."""
    archs = archetype_computer(
        adata.X.transpose(),
        w=w,
        s=s,
    )

    adata.obsm["archetypes"] = archs.toarray().transpose()


def archetype_computer(
    data,
    w=None,
    s=None,
):
    """
    Compute archetype scores from expression data.

    Parameters
    ----------
    data
        Gene expression matrix.
    w
        Pretrained archetype weight matrix.
    s
        Gene scaling factors.

    Returns
    -------
    h
        Archetype score matrix.
    """
    # Load pretrained archetype weights if not supplied.
    if w is None:
        wlist = []

        with Path(
            "/Users/kocherc/Documents/GitHub/code/MATLAB/"
            "Cancer_Archetypes-main/Outputs/Archetypes.csv"
        ).open() as file:
            csv_reader = csv.reader(file)

            wlist.extend([float(value) for value in row] for row in csv_reader)

        w = csr_matrix(np.array(wlist))

    # Load pretrained gene scaling factors if not supplied.
    if s is None:
        slist = []

        with Path(
            "/Users/kocherc/Documents/GitHub/code/MATLAB/"
            "Cancer_Archetypes-main/Outputs/S.csv"
        ).open() as file:
            csv_reader = csv.reader(file)

            slist.extend(float(row[0]) for row in csv_reader)

        s = np.array(slist)

    # Reweight genes.
    dd = data / s[:, None]

    # Normalize each observation.
    dd = dd / dd.sum(axis=0)

    return nmf_new_weights(
        dd,
        w,
        6,
    )


def nmf_new_weights(
    v,
    w,
    k,
):
    """
    Compute archetype scores using pretrained weights.

    Parameters
    ----------
    v
        Input expression matrix.
    w
        Pretrained archetype weight matrix.
    k
        Number of archetypes.

    Returns
    -------
    h
        Archetype score matrix.
    """
    n_iter = 1000

    # Fixed random seed for reproducibility.
    rng = np.random.default_rng(0)

    eps = np.finfo(float).eps

    _, n = v.shape

    # Initialize archetype contribution matrix.
    h = csr_matrix(rng.random((k, n)) + eps)

    # Multiplicative NMF update while keeping W fixed.
    for _ in range(n_iter):
        wv = w.transpose().dot(v)
        wwh = w.transpose().dot(w.dot(h))

        h = h.multiply(wv / wwh)

        # Normalize archetype contributions.
        h = h / h.sum(axis=0)

    return h


def add_archetypes_to_adata(
    adata: AnnData,
    w=None,
    s=None,
):
    """
    Add archetype scores to an AnnData object.

    Stores results in adata.obsm["archetypes"].
    """
    archs = archetype_computer(
        adata.X.transpose(),
        w=w,
        s=s,
    )

    adata.obsm["archetypes"] = archs.toarray().transpose()


def compute_hexagon_coordinates(
    archetypes,
):
    """Convert archetype scores into two-dimensional coordinates."""
    # Order archetypes according to the hexagon visualization layout.
    hh = archetypes[:, [1, 2, 0, 4, 3, 5]]

    directions = np.array([0, 1, 2, 3, 4, 5])

    direction_x = np.cos(np.pi / 2 - 2 * np.pi * directions / 6)

    direction_y = np.sin(np.pi / 2 - 2 * np.pi * directions / 6)

    coor1 = hh.dot(direction_x)
    coor2 = hh.dot(direction_y)

    return coor1, coor2


def add_coor_to_adata(
    adata: AnnData,
):
    """
    Add hexagon coordinates to an AnnData object.

    Stores coordinates in adata.obsm["Coor1"] and adata.obsm["Coor2"].
    """
    coor1, coor2 = compute_hexagon_coordinates(adata.obsm["archetypes"])

    adata.obsm["Coor1"] = coor1
    adata.obsm["Coor2"] = coor2
