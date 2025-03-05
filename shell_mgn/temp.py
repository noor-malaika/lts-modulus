from utils import save_test_idx, load_test_idx
import torch
idx = [
    ('var-54', '90'),
    ('var-78', '56')
]

save_test_idx(idx)

print(load_test_idx())