"""Deterministic sample-wise history permutation during both training and evaluation."""
import numpy as np
import run_6306_levels as runner
original = runner.transform
def shuffled(x, scales, selected):
    z = original(x, scales, selected)
    # Content-derived seed is independent of labels, phase, batch and model seed.
    import hashlib
    seed = int.from_bytes(hashlib.sha256(np.ascontiguousarray(x).tobytes()).digest()[:8], 'little')
    return z[np.random.default_rng(seed).permutation(len(z))]
runner.transform = shuffled
base_identity = runner.code_identity
def identity():
    result = base_identity()
    result['run_6306_shuffled.py'] = runner.digest(__file__)
    return result
runner.code_identity = identity
if __name__ == '__main__':
    runner.main()
