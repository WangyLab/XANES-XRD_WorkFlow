import numpy as np
from pymatgen.core import Structure
from scipy.spatial.distance import cdist

def kabsch_rmsd(P, Q):
    assert P.shape == Q.shape, "The structures must have the same number of atoms!"

    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = np.dot(P_centered.T, Q_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(U, Vt)

    Q_rotated = np.dot(Q_centered, R)
    rmsd = np.sqrt(np.mean(np.sum((P_centered - Q_rotated) ** 2, axis=1)))

    return rmsd

structure1 = Structure.from_file("A.cif")
structure2 = Structure.from_file("B.cif")

structure1.apply_strain(0)
structure2.apply_strain(0)

coords1 = np.array(structure1.frac_coords)
coords2 = np.array(structure2.frac_coords)

dist_matrix = cdist(coords1, coords2)
min_indices = np.argmin(dist_matrix, axis=1)
coords2_sorted = coords2[min_indices]

rmsd = kabsch_rmsd(coords1, coords2_sorted)
print(f"RMSD between the structures (ignoring lattice parameters): {rmsd:.4f} Å")
