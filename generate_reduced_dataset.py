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

    targets = targets.reshape(targets.shape[0],1)

    new = np.concatenate((inputs,targets),axis=1)
    new = new[new[:,3072].argsort()]

    num_entries = num

    curr_class = 0
    curr_class_count = 0
    index_array = []
    for i in range(new.shape[0]):
        if new[i,-1] != curr_class:
            continue
        if curr_class_count < num_entries:
            index_array.append(i)
            curr_class_count += 1
        else:
            curr_class += 1
            curr_class_count = 0

    okastren = new[index_array]

    okastren_i_oformen = okastren[:,0:-1].reshape(100,300,-1)
    np.save(fn, okastren_i_oformen)
    print("Saving in: ", fn)
else:
    print("Provide number of samples and filename like:")
    print("python generate_reduced_dataset.py 20 cifar-100-20.npy")
