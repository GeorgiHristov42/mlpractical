from one_shot_learning_network import *
from experiment_builder import ExperimentBuilder
import tensorflow.contrib.slim as slim
import data as dataset
import tqdm
from storage import save_statistics, build_experiment_folder

tf.reset_default_graph()

# Experiment Setup
batch_size = 32
fce = False
classes_per_set = 3
samples_per_class = 1
continue_from_epoch = -1  # use -1 to start from scratch
transfer_enabled = 10 # change to anything else than -1 to use another saved model
transfer_from_path = 'cifar100_one_shot_learning_embedding_1_10/saved_models/cifar100_one_shot_learning_embedding_1_10_20.ckpt'
epochs = 50
num_gpus = 1
logs_path = "one_shot_outputs/"
experiment_name = "cifar100_to_10_one_shot_learning_embedding_{}_{}".format(samples_per_class, classes_per_set)

# Experiment builder
data = dataset.FolderDatasetLoader(num_of_gpus=num_gpus, batch_size=batch_size, image_height=32, image_width=32,
                                   image_channels=3,
                                   train_val_test_split=(6/10, 3/10, 0/10),
                                   samples_per_iter=1, num_workers=4,
                                   data_path="datasets/cifar10_cut", name="cifar10_cut",
                                   index_of_folder_indicating_class=-2, reset_stored_filepaths=False,
                                   num_samples_per_class=samples_per_class, num_classes_per_set=classes_per_set)

experiment = ExperimentBuilder(data)
one_shot_omniglot, losses, c_error_opt_op, init = experiment.build_experiment(batch_size,
                                                                                     classes_per_set,
                                                                                     samples_per_class, fce)
total_epochs = 50
total_train_batches = 500
total_val_batches = 100
total_test_batches = 100

saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name)


save_statistics(logs_filepath, ["epoch", "total_train_c_loss_mean", "total_train_c_loss_std",
                                  "total_train_accuracy_mean", "total_train_accuracy_std", "total_val_c_loss_mean",
                                  "total_val_c_loss_std", "total_val_accuracy_mean", "total_val_accuracy_std",
                                  "total_test_c_loss_mean", "total_test_c_loss_std", "total_test_accuracy_mean",
                                  "total_test_accuracy_std"], create=True)

# Experiment initialization and running
with tf.Session() as sess:
    sess.run(init)
    train_saver = tf.train.Saver()
    val_saver = tf.train.Saver()
    if transfer_enabled != -1:
        print('TRANSFER TESTING')
        checkpoint = transfer_from_path

        variables_to_restore = []
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(var)
            variables_to_restore.append(var)
        tf.logging.info('Fine-tuning from %s' % checkpoint)

        fine_tune = slim.assign_from_checkpoint_fn(
            checkpoint,
            variables_to_restore,
            ignore_missing_vars=True)
        fine_tune(sess)

    if continue_from_epoch != -1: #load checkpoint if needed
        print('TESTING')
        checkpoint = "{}/{}_{}.ckpt".format(saved_models_filepath,experiment_name, continue_from_epoch)
        variables_to_restore = []
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(var)
            variables_to_restore.append(var)

        tf.logging.info('Fine-tuning from %s' % checkpoint)

        fine_tune = slim.assign_from_checkpoint_fn(
            checkpoint,
            variables_to_restore,
            ignore_missing_vars=True)
        fine_tune(sess)

    best_val_acc_mean = 0.
    best_val_epoch = 6
    with tqdm.tqdm(total=total_epochs) as pbar_e:
        for e in range(continue_from_epoch+1, total_epochs):
            total_train_c_loss_mean, total_train_c_loss_std, total_train_accuracy_mean, total_train_accuracy_std =\
                experiment.run_training_epoch(total_train_batches=total_train_batches,
                                                                                sess=sess)
            print("Epoch {}: train_loss_mean: {}, train_loss_std: {}, train_accuracy_mean: {}, train_accuracy_std: {}"
                  .format(e, total_train_c_loss_mean, total_train_c_loss_std, total_train_accuracy_mean, total_train_accuracy_std))

            total_val_c_loss_mean, total_val_c_loss_std, total_val_accuracy_mean, total_val_accuracy_std = \
                experiment.run_validation_epoch(total_val_batches=total_val_batches,
                                              sess=sess)
            print("Epoch {}: val_loss_mean: {}, val_loss_std: {}, val_accuracy_mean: {}, val_accuracy_std: {}"
                  .format(e, total_val_c_loss_mean, total_val_c_loss_std, total_val_accuracy_mean,
                          total_val_accuracy_std))

            if total_val_accuracy_mean >= best_val_acc_mean: #if new best val accuracy -> produce test statistics
                best_val_acc_mean = total_val_accuracy_mean
                best_val_epoch = e

                val_save_path = val_saver.save(sess, "{}/best_val_{}_{}.ckpt".format(saved_models_filepath, experiment_name, e))

                total_test_c_loss_mean, total_test_c_loss_std, total_test_accuracy_mean, total_test_accuracy_std \
                    = -1, -1, -1, -1

            save_statistics(logs_filepath,
                            [e, total_train_c_loss_mean, total_train_c_loss_std, total_train_accuracy_mean,
                             total_train_accuracy_std, total_val_c_loss_mean, total_val_c_loss_std,
                             total_val_accuracy_mean, total_val_accuracy_std,
                             total_test_c_loss_mean, total_test_c_loss_std, total_test_accuracy_mean,
                             total_test_accuracy_std])

            save_path = train_saver.save(sess, "{}/{}_{}.ckpt".format(saved_models_filepath, experiment_name, e))
            pbar_e.update(1)

        val_saver.restore(sess,
                          "{}/best_val_{}_{}.ckpt".format(saved_models_filepath, experiment_name, best_val_epoch))
        total_test_c_loss_mean, total_test_c_loss_std, total_test_accuracy_mean, total_test_accuracy_std = \
            experiment.run_testing_epoch(total_test_batches=total_test_batches,
                                         sess=sess)
        print(
            "Epoch {}: test_loss_mean: {}, test_loss_std: {}, test_accuracy_mean: {}, test_accuracy_std: {}"
                .format(best_val_epoch, total_test_c_loss_mean, total_test_c_loss_std, total_test_accuracy_mean,
                        total_test_accuracy_std))
        save_statistics(logs_filepath,
                        ["Test error on best validation model", -1, -1, -1,
                         -1, -1, -1,
                         -1, -1,
                         total_test_c_loss_mean, total_test_c_loss_std, total_test_accuracy_mean,
                         total_test_accuracy_std])
