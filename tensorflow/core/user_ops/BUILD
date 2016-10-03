load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "py_tests")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_tests")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

tf_custom_op_library(
    name = "locally_connected.so",
    srcs = ["locally_connected.cc",
            "//tensorflow/core/kernels:transpose_functor.h"],
    )

py_tests(
    name = "locally_connected_test",
    size = "small",
    srcs = ["locally_connected_test.py"],
    data = [":locally_connected.so"]
    # can't figure out how to make ^^^ work, so using "external" instead to
    # prevent caching
    # tags = ["external"],
)


