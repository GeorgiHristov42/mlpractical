import sys
import numpy as np
import os

if len(sys.argv) == 3:
    num = int(sys.argv[1])
    fn = sys.argv[2]

    data_path = os.path.join(
            os.environ['MLP_DATA_DIR'], 'cifar100-train.npz')

    loaded = np.load(data_path)

    inputs, targets = loaded['inputs'], loaded['targets']
    inputs = inputs.astype(np.float32)

    num_entries = num

    n_of_class_incl = np.zeros(100)

    index_array = []
    for i in range(targets.shape[0]):
        if n_of_class_incl[targets[i]] > num_entries - 1:
            continue
        else:
            n_of_class_incl[targets[i]] += 1
            index_array.append(i)

    okastren_targets = targets[index_array]
    okastren_inputs = inputs[index_array]

    save_data = {'inputs': okastren_inputs, 'targets': okastren_targets}

    np.save(fn, save_data)
    print("Saving in: ", fn)
else:
    print("Provide number of samples and filename like:")
    print("python reduce_size_keep_format.py 20 cifar-100-20.npy")
