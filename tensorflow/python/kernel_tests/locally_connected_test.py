#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import random
import math

# a single 4x4 image with 1 channel
t = [[
  [[1.], [2.], [3.], [4.]],
  [[5.], [6.], [7.], [8.]],
  [[9.], [10.], [11.], [12.]],
  [[13.], [14.], [15.], [16.]]
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

class LocConnTest(tf.test.TestCase):
  def testLocConn(self):
    loc_conn_module = tf.load_op_library("/Users/john/arpg/tensorflow/" +
        "bazel-bin/tensorflow/core/user_ops/locally_connected.so")
    with self.test_session():
      # a test I can wrap my head around
      result = loc_conn_module.loc_conn(t, f)
      self.assertAllEqual(result.eval(), [
        [[6.], [6.]],
        [[6.], [6.]]])

      # a test I cannot
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
                    random.rand() + 0.1
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
                random.rand() * 10 - 5
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

      h_stride = (in_height - filter_height + 1)/out_height
      w_stride = (in_width - filter_width + 1)/out_width
      h_pos = [math.ceil(h_stride * i) for i in range(out_height)]
      w_pos = [math.ceil(w_stride * i) for i in range(out_width)]

      for b in range(batch):
        for i in range(out_height):
          for j in range(out_width):
            for k in range(out_channels):
              # apply the appropriate filter
              for f_i in range(out_height):
                for f_j in range(out_width):
                  for f_k in range(in_channels):
                    out_tensor[b][i][j][k] += (f_tensor[i][j][f_i][f_j][f_k][k] *
                        in_tensor[b][f_i + h_pos[i]][f_j + w_pos[j]][f_k])

      # run test
      result = loc_conn_module.loc_conn(in_tensor, f_tensor)
      self.assertAllEqual(result.eval(), out_tensor)

  def testLocConnGrad(self):
    with self.test_session():
      self.assertAllEqual(1, 1)
