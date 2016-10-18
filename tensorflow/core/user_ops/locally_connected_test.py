#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
import random
import math
import numpy as np


# a single 4x4 image with 1 channel
t = [[
  [[ 1.], [ 2.], [ 3.], [ 4.]],
  [[ 5.], [ 6.], [ 7.], [ 8.]],
  [[ 9.], [10.], [11.], [12.]],
  [[13.], [14.], [15.], [16.]]
  ]]
t_shape = [1, 4, 4, 1]

g_t = [[
  [[0.], [0.], [0.], [0.]],
  [[0.], [1.], [0.], [0.]],
  [[0.], [0.], [0.], [0.]],
  [[0.], [0.], [0.], [0.]]
  ]]

# a 2x2 array of 3x3x1x1 filters
f = [
    [
      [
        [[[0.]], [[0.]], [[0.]]],
        [[[0.]], [[1.]], [[0.]]],
        [[[0.]], [[0.]], [[0.]]]
        ],
      [
        [[[0.]], [[0.]], [[0.]]],
        [[[1.]], [[0.]], [[0.]]],
        [[[0.]], [[0.]], [[0.]]]
        ]
      ],
    [
      [
        [[[0.]], [[1.]], [[0.]]],
        [[[0.]], [[0.]], [[0.]]],
        [[[0.]], [[0.]], [[0.]]]
        ],
      [
        [[[1.]], [[0.]], [[0.]]],
        [[[0.]], [[0.]], [[0.]]],
        [[[0.]], [[0.]], [[0.]]]
        ]
      ]
    ]
f_shape = [2, 2, 3, 3, 1, 1]

g_f = [
    [
      [
        [[[ 1.]], [[ 2.]], [[ 3.]]],
        [[[ 5.]], [[ 6.]], [[ 7.]]],
        [[[ 9.]], [[10.]], [[11.]]]
        ],
      [
        [[[- 2.]], [[- 3.]], [[- 4.]]],
        [[[- 6.]], [[- 7.]], [[- 8.]]],
        [[[-10.]], [[-11.]], [[-12.]]]
        ]
      ],
    [
      [
        [[[- 5.]], [[- 6.]], [[- 7.]]],
        [[[- 9.]], [[-10.]], [[-11.]]],
        [[[-13.]], [[-14.]], [[-15.]]]
        ],
      [
        [[[12.]], [[14.]], [[16.]]],
        [[[20.]], [[22.]], [[24.]]],
        [[[28.]], [[30.]], [[32.]]]
        ]
      ]
    ]

g = [[
  [[ 1.], [-1.]],
  [[-1.], [ 2.]]]]

def create_random_test():
  batch = 4
  in_height = 8
  in_width = 12
  in_channels = 3
  filter_height = 3
  filter_width = 3
  out_height = 4
  out_width = 6
  out_channels = 2

  # create filter
  f_tensor = [
      [
        [
          [
            [
              [
                random.random() + 0.1
                for _ in range(out_channels)]
              for _ in range(in_channels)]
            for _ in range(filter_width)]
          for _ in range(filter_height)]
        for _ in range(out_width)]
      for _ in range(out_height)]

  # create input tensor
  in_tensor = [
      [
        [
          [
            random.random() * 10 - 5
            for _ in range(in_channels)]
          for _ in range(in_width)]
        for _ in range(in_height)]
      for _ in range(batch)]

  # calculate output tensor
  out_tensor = [
      [
        [
          [
            0
            for _ in range(out_channels)]
          for _ in range(out_width)]
        for _ in range(out_height)]
      for _ in range(batch)]

  h_stride = (in_height - filter_height + 1.)/out_height
  w_stride = (in_width - filter_width + 1.)/out_width
  h_pos = [int(math.ceil(h_stride * i)) for i in range(out_height)]
  w_pos = [int(math.ceil(w_stride * i)) for i in range(out_width)]

  for b in range(batch):
    for i in range(out_height):
      for j in range(out_width):
        for k in range(out_channels):
          # apply the appropriate filter
          for f_i in range(filter_height):
            for f_j in range(filter_width):
              for f_k in range(in_channels):
                out_tensor[b][i][j][k] += (f_tensor[i][j][f_i][f_j][f_k][k] *
                    in_tensor[b][f_i + h_pos[i]][f_j + w_pos[j]][f_k])
  return (tf.constant(f_tensor), tf.constant(in_tensor), tf.constant(out_tensor))

loc_conn_module = tf.load_op_library("/Users/john/arpg/tensorflow/" +
    "bazel-bin/tensorflow/core/user_ops/locally_connected.so")

class LocConnTest(tf.test.TestCase):
  @ops.RegisterGradient("LocConn")
  def _loc_conn_grad(op, grad):
    return loc_conn_module.loc_conn_grad(grad, op.inputs[0], op.inputs[1])

  def testLocConnSimple(self):
    '''A simple test of the locally connected layer'''
    with self.test_session():
      result = loc_conn_module.loc_conn(t, f)
      self.assertAllEqual(result.eval(), [[
        [[6.], [6.]],
        [[6.], [6.]]]])

  def testLocConnRandom(self):
    '''A complicated, randomized test of the locally connected layer'''
    with self.test_session():
      (f_tensor, in_tensor, out_tensor) = create_random_test()

      # run test
      result = loc_conn_module.loc_conn(in_tensor, f_tensor)
      self.assertAllClose(result.eval(), out_tensor.eval())

  def testLocConnGradSimple(self):
    '''A simple test of the locally connected layer gradient'''
    with self.test_session():
      result = loc_conn_module.loc_conn_grad(g, t, f)
      self.assertAllEqual(result[0].eval(), g_t)
      self.assertAllEqual(result[1].eval(), g_f)

    with self.test_session():
      in_tensor = tf.constant(t)
      f_tensor = tf.constant(f)
      result = loc_conn_module.loc_conn(in_tensor, f_tensor)
      o_shape = [1, 2, 2, 1]
      grads = tf.test.compute_gradient(
          [f_tensor, in_tensor], [f_shape, t_shape], result, o_shape)
      self.assertAllClose(grads[0][0], grads[0][1], rtol=1e-3, atol=1e-3)
      self.assertAllClose(grads[1][0], grads[1][1], rtol=1e-3, atol=1e-3)

  def testLocConnGradRandom(self):
    '''A complicated, randomized test of the locally connected layer gradient'''
    with self.test_session():
      (f_tensor, in_tensor, out_tensor) = create_random_test()
      result = loc_conn_module.loc_conn(in_tensor, f_tensor)
      f_shape = [int(d) for d in f_tensor.get_shape()]
      i_shape = [int(d) for d in in_tensor.get_shape()]
      o_shape = [int(d) for d in out_tensor.get_shape()]
      grads = tf.test.compute_gradient(
          [f_tensor, in_tensor], [f_shape, i_shape], result, o_shape)
      self.assertAllClose(grads[0][0], grads[0][1], rtol=1e-3, atol=1e-3)
      self.assertAllClose(grads[1][0], grads[1][1], rtol=1e-3, atol=1e-3)

  def testLocConnTrainVerySimple(self):
    '''very simple training test'''
    input_size = [5, 4]
    output_size = [3, 2]
    filter_size = [3, 3]
    batch_size = 1000
    total_batches = 1000

    def f(i_img):
      '''very simple function we're trying to learn'''
      o_img = (
          i_img[0:output_size[0], 0:output_size[1]] +
          i_img[input_size[0]-output_size[0]:, 0:output_size[1]] +
          i_img[0:output_size[0], input_size[1]-output_size[1]:] +
          i_img[input_size[0]-output_size[0]:, input_size[1]-output_size[1]:])
      return o_img

    out_t = np.zeros([output_size[0], output_size[1], filter_size[0],
      filter_size[1], 1, 1])
    for i in range(output_size[0]):
      for j in range(output_size[1]):
        for fi in range(filter_size[0]):
          for fj in range(filter_size[1]):
            if fi in (0, 2) and fj in (0, 2):
              out_t[i][j][fi][fj][0][0] = 1.

    with self.test_session():
      x = tf.placeholder(tf.float32, shape=(None, input_size[0], input_size[1], 1))
      filter_t = tf.Variable(tf.truncated_normal(
        [output_size[0], output_size[1], filter_size[0], filter_size[1], 1, 1],
        stddev=0.2))
      y = loc_conn_module.loc_conn(x, filter_t)

      y_ = tf.placeholder(tf.float32, shape=(None, output_size[0], output_size[1], 1))

      sqrerr = tf.reduce_mean(tf.squared_difference(y, y_))
      train_step = tf.train.GradientDescentOptimizer(0.5).minimize(sqrerr)

      tf.initialize_all_variables().run()

      for i in range(total_batches):
        batch_xs = np.random.rand(batch_size, input_size[0], input_size[1], 1)
        batch_ys = np.asarray([f(xs) for xs in batch_xs])
        #batch_ys = np.expand_dims(batch_ys, 3)
        train_step.run({x:batch_xs, y_:batch_ys})
      self.assertAllClose(filter_t.eval(), out_t, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
  tf.test.main()
