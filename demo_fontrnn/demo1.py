import numpy as np
import os
import matplotlib.pyplot as plt
# import sys
# import
import train
import model
import utils
import gmm
import tensorflow as tf
import random

from model import FontRNN, get_default_hparams, copy_hparams


model_dir = '../log/demo'
data_filepath = '../data/FZTLJW_775.npz'


def getinfo():

    testing_mode = True
    # a = np.load('../data/FZTLJW_775.npz')['test']
    data = np.load(data_filepath, allow_pickle=True, encoding='latin1')
    # target data
    train_strokes = data['train']
    valid_strokes = data['valid']
    test_strokes = data['test']
    all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))

    # standard data (reference data in paper)
    std_train_strokes = data['std_train']
    std_valid_strokes = data['std_valid']
    std_test_strokes = data['std_test']
    all_std_trokes = np.concatenate((std_train_strokes, std_valid_strokes, std_test_strokes))

    print('Dataset combined: %d (train=%d/validate=%d/test=%d)' % (
        len(all_strokes), len(train_strokes), len(valid_strokes), len(test_strokes)))
    # calculate the max strokes we need.
    max_seq_len = utils.get_max_len(all_strokes)
    max_std_seq_len = utils.get_max_len(all_std_trokes)
    # overwrite the hps with this calculation.
    model_params = get_default_hparams()
    model_params.max_seq_len = max(max_seq_len, max_std_seq_len)
    print('model_params.max_seq_len set to %d.' % model_params.max_seq_len)

    eval_model_params = copy_hparams(model_params)
    eval_model_params.rnn_dropout_keep_prob = 1.0
    eval_model_params.is_training = True

    if testing_mode:  # for testing
        eval_model_params.batch_size = 1
        eval_model_params.is_training = False  # sample mode

    train_set = utils.DataLoader(
            train_strokes,
            model_params.batch_size,
            max_seq_length=model_params.max_seq_len,
            random_scale_factor=model_params.random_scale_factor,
            augment_stroke_prob=model_params.augment_stroke_prob)
    normalizing_scale_factor = model_params.scale_factor
    # print(normalizing_scale_factor)
    train_set.normalize(normalizing_scale_factor)
    # print(train_set)

    valid_set = utils.DataLoader(
        valid_strokes,
        eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    valid_set.normalize(normalizing_scale_factor)

    test_set = utils.DataLoader(
        test_strokes,
        eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    test_set.normalize(normalizing_scale_factor)

    # process the reference dataset
    std_train_set = utils.DataLoader(
        std_train_strokes,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor,
        augment_stroke_prob=model_params.augment_stroke_prob)
    std_train_set.normalize(normalizing_scale_factor)

    std_valid_set = utils.DataLoader(
        std_valid_strokes,
        eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    std_valid_set.normalize(normalizing_scale_factor)

    std_test_set = utils.DataLoader(
        std_test_strokes,
        eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    std_test_set.normalize(normalizing_scale_factor)

    result = [
        train_set, valid_set, test_set,
        std_train_set, std_valid_set, std_test_set,
        model_params, eval_model_params
    ]
    return result


def test_model(sess, testmodel, input_stroke):
    stroke_len = len(input_stroke)
    input_stroke = utils.to_big_strokes(input_stroke, max_len=testmodel.hps.max_seq_len).tolist()
    input_stroke.insert(0, [0, 0, 1, 0, 0])
    feed = {testmodel.enc_input_data: [input_stroke],
            testmodel.enc_seq_lens: [stroke_len],
            }
    output = sess.run([testmodel.pi, testmodel.mu1, testmodel.mu2, testmodel.sigma1,
                       testmodel.sigma2, testmodel.corr, testmodel.pen,
                       testmodel.timemajor_alignment_history],
                      feed)
    # print(output)
    # print(len(output))
    gmm_params = output[:-1]
    timemajor_alignment_history = output[7]

    return gmm_params, timemajor_alignment_history


def sample_from_params(params, temp=0.1, greedy=False):
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen] = params

    max_len = o_pi.shape[0]
    print(max_len)
    num_mixture = o_pi.shape[1]
    print(num_mixture)
    strokes = np.zeros((max_len, 5), dtype=np.float32)

    for step in range(max_len):
        next_x1 = 0
        next_x2 = 0
        eos = [0, 0, 0]
        eos[np.argmax(o_pen[step])] = 1
        for mixture in range(num_mixture):
            x1, x2 = gmm.sample_gaussian_2d(o_mu1[step][mixture], o_mu2[step][mixture],
                                            o_sigma1[step][mixture], o_sigma2[step][mixture],
                                            o_corr[step][mixture], np.sqrt(temp), greedy)
            next_x1 += x1 * o_pi[step][mixture]
            next_x2 += x2 * o_pi[step][mixture]
        strokes[step, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]
    strokes = utils.to_normal_strokes(strokes)
    return strokes


def draw(delta_gt_stroke, delta_stroke):
    ground_truth_stroke = delta_gt_stroke.copy()
    stroke = delta_stroke.copy()

    # convert to absolute coordinate
    scale_factor = 300
    low_tri_matrix = np.tril(np.ones((delta_gt_stroke.shape[0], delta_gt_stroke.shape[0])), 0)
    # print('low_tri_matrix: ',low_tri_matrix)
    # print('low_tri_matrix: ',len(low_tri_matrix))
    # print('low_tri_matrix: ',low_tri_matrix.shape)
    # print(np.ones((delta_gt_stroke.shape[0], delta_gt_stroke.shape[0])))
    # print(np.ones((delta_gt_stroke.shape[0], delta_gt_stroke.shape[0])).shape)
    # print(delta_gt_stroke.shape)
    print(type(ground_truth_stroke))
    ground_truth_stroke[:, :2] = np.rint(scale_factor * np.matmul(low_tri_matrix, delta_gt_stroke[:, :2]))
    print(ground_truth_stroke)
    low_tri_matrix = np.tril(np.ones((delta_stroke.shape[0], delta_stroke.shape[0])), 0)
    stroke[:, :2] = np.rint(scale_factor * np.matmul(low_tri_matrix, delta_stroke[:, :2]))

    plt.figure(figsize=(6, 3))
    plt.subplot(121)
    # plt.xlim(0, 300)
    # plt.ylim(0, 300)
    pre_i = 0
    print( ground_truth_stroke[0:1+1,0])
    print( ground_truth_stroke[0:1+1,1])
    for i in range(ground_truth_stroke.shape[0]):
        if ground_truth_stroke[i][2] == 1:
            plt.plot(ground_truth_stroke[pre_i:i + 1, 0], ground_truth_stroke[pre_i:i + 1, 1], color='black',
                     linewidth=3)
            pre_i = i + 1
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.title("Groud Truth")

    plt.subplot(122)
    # plt.xlim(0, 300)
    # plt.ylim(0, 300)
    pre_i = 0
    for i in range(stroke.shape[0]):
        if stroke[i][2] == 1:
            plt.plot(stroke[pre_i:i + 1, 0], stroke[pre_i:i + 1, 1], color='black', linewidth=3)
            pre_i = i + 1
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.title("Generated")

    plt.show()

# print(result[0])
# print(result[6])
# print(result[7])
[train_set, valid_set, test_set, std_train_set, std_valid_set, std_test_set,
     hps_model, eval_hps_model]  = getinfo()
# print('1111111111111111111111111111111111111',eval_hps_model)
train.reset_graph()
train_model = model.FontRNN(hps_model)
eval_model = model.FontRNN(eval_hps_model, reuse=True)

# print(hps_model)
# print(eval_hps_model)

# load trained checkpoint
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
train.load_checkpoint(sess, model_dir)

index = random.randint(0, len(std_test_set.strokes)-1)

ref_strokes = np.copy(std_test_set.strokes[index])
real_strokes = np.copy(test_set.strokes[index])
print(ref_strokes)
# print(index)
# print('test_set.strokes',type(test_set.strokes))
# print('len:',len(test_set.strokes))
# print('ref_strokes:',std_test_set)
# print('real_strokes:',ref_strokes[0])

params, timemajor_alignment_history = test_model(sess, eval_model, ref_strokes)
# print(len(params))
# print(params[0])
fake_strokes = sample_from_params(params, greedy=True)
# print(fake_strokes)

draw(real_strokes,fake_strokes)
plt.show()