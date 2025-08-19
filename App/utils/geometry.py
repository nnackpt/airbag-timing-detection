import numpy as np

def is_far_enough(new_center, centers, min_dist=45):
    new_center = np.array(new_center, dtype=np.float32)
    for c in centers:
        c = np.array(c, dtype=np.float32)
        if np.linalg.norm(new_center - c) < min_dist:
            return False
    return True