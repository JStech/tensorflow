#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/transpose_functor.h"

namespace tensorflow {

  void verify_dims(OpKernelContext* const context, const Tensor& input,
      const Tensor& filter, const Tensor* const gradients) {
    // check dimensions
    OP_REQUIRES(context, input.dims() == 4,
        errors::InvalidArgument("input must be 4-dimensional",
          input.shape().DebugString()));

    OP_REQUIRES(context, filter.dims() == 6,
        errors::InvalidArgument("filter must be 6-dimensional",
          filter.shape().DebugString()));

    OP_REQUIRES(
        context, input.dim_size(3) == filter.dim_size(4),
        errors::InvalidArgument("input and filter must have the same depth: ",
          input.dim_size(3), " vs ", filter.dim_size(4)));

    if (gradients != nullptr) {
      OP_REQUIRES(context, gradients->dims() == 4,
          errors::InvalidArgument("gradients must be 4-dimensional",
            input.shape().DebugString()));

      OP_REQUIRES( context, input.dim_size(0) == gradients->dim_size(0),
          errors::InvalidArgument("input and gradients must have the same batch size: ",
            input.dim_size(0), " vs ", gradients->dim_size(0)));

      OP_REQUIRES(context,
          filter.dim_size(0) == gradients->dim_size(1) &&
          filter.dim_size(1) == gradients->dim_size(2) &&
          filter.dim_size(5) == gradients->dim_size(3),
          errors::InvalidArgument("height, width, and depth of filter and gradients must match: ",
            filter.dim_size(0), " vs ", gradients->dim_size(1), " and ",
            filter.dim_size(1), " vs ", gradients->dim_size(2), " and ",
            filter.dim_size(5), " vs ", gradients->dim_size(3)));
    }
  }

REGISTER_OP("LocConn")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {half, float, double} = DT_FLOAT")
    //.SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
Computes a 2-D locally connected layer given 4-D `input` and 6-D `filter` tensors.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter tensor of shape `[out_height, out_width, filter_height,
filter_width, in_channels, out_channels]`, this op calculates an output tensor
of shape `[batch, out_height, out_width, out_channels]` where each element is
calculated as follows,

    output[b, i, j, c] = sum over f_i, f_j, f_c (
      filter[i, j, f_i, f_j, f_c, c] * input[b, pos_i+f_i, pos_j+f_j, f_c])

The `pos_i` and `pos_j` values are rounded multiples of a potentially
non-integer stride. The filters at the four corners of the output are
positioned so that they cover the respective corners of the input, and the
remaining filters are distributed evenly between them. There is no padding of
the input (as is common with a convolutional layer).
)doc");

class LocConnOp : public OpKernel {
  public:
    explicit LocConnOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      const Tensor& input = context->input(0);
      const Tensor& filter = context->input(1);

      verify_dims(context, input, filter, nullptr);

      const int64 in_height = input.dim_size(1);
      const int64 in_width = input.dim_size(2);
      const int64 in_depth = input.dim_size(3);

      // output dimensions
      const int64 batch = input.dim_size(0);
      const int out_height = static_cast<int>(filter.dim_size(0));
      const int out_width = static_cast<int>(filter.dim_size(1));
      const int out_depth = static_cast<int>(filter.dim_size(5));

      // filter dimensions
      const int filter_height = static_cast<int>(filter.dim_size(2));
      const int filter_width = static_cast<int>(filter.dim_size(3));

      // flatten filter over filter height/width and input channels
      Tensor* flat_filter;

      // Create an output tensor
      Tensor* output = NULL;
      std::vector<int64> dim_sizes(4);
      dim_sizes[0] = batch;
      dim_sizes[1] = out_height;
      dim_sizes[2] = out_width;
      dim_sizes[3] = out_depth;
      TensorShape out_shape = TensorShape(dim_sizes);
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
            &output));

    VLOG(2) << "LocConn: in_depth = " << in_depth
            << ", out_depth = " << out_depth;

      if (out_shape.num_elements() == 0) {
        return;
      }

      float h_stride = (in_height - filter_height + 1.)/out_height;
      float w_stride = (in_width - filter_width + 1.)/out_width;

      for (int b=0; b<batch; b++) {
        for (int k=0; k<out_depth; k++) {
          for (int i=0; i<out_height; i++) {
            for (int j=0; j<out_width; j++) {
              double sum = 0.;
              for (int fi=0; fi<filter_height; fi++) {
                for (int fj=0; fj<filter_width; fj++) {
                  int in_i = h_stride * i + fi;
                  int in_j = w_stride * j + fj;
                  for (int fk=0; fk<in_depth; fk++) {
                    sum += filter.tensor<float, 6>()(i, j, fi, fj, fk, k) * input.tensor<float, 4>()(b, in_i, in_j, fk);
                  }
                }
              }
              output->tensor<float, 4>()(b, i, j, k) = sum;
            }
          }
        }
      }
    }
};

REGISTER_OP("LocConnGrad")
    .Input("gradients: T")
    .Input("input: T")
    .Input("filter: T")
    .Output("gradient_input: T")
    .Output("gradient_filter: T")
    .Attr("T: {half, float, double} = DT_FLOAT")
    //.SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
Gradient calculation for LocConn.
)doc");

class LocConnGradOp : public OpKernel {
  public:
    explicit LocConnGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // get inputs
      const Tensor& gradients = context->input(0);
      const Tensor& input = context->input(1);
      const Tensor& filter = context->input(2);

      verify_dims(context, input, filter, &gradients);

      // confusing: one of the outputs of this function is gradient_input (dL/dinput)
      Tensor* gradient_input = NULL;
      Tensor* gradient_filter = NULL;

      TensorShape input_shape = input.shape();
      TensorShape filter_shape = filter.shape();

      OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &gradient_input));
      OP_REQUIRES_OK(context, context->allocate_output(1, filter_shape, &gradient_filter));

      const int64 batch_size = input.dim_size(0);
      const int64 in_height = input.dim_size(1);
      const int64 in_width = input.dim_size(2);
      const int64 out_height = filter.dim_size(0);
      const int64 out_width = filter.dim_size(1);
      const int64 filter_height = filter.dim_size(2);
      const int64 filter_width = filter.dim_size(3);
      const int64 in_channels = filter.dim_size(4);
      const int64 out_channels = filter.dim_size(5);

      float h_stride = (in_height - filter_height + 1.)/out_height;
      float w_stride = (in_width - filter_width + 1.)/out_width;

      for (int b=0; b<batch_size; b++) {
        for (int i=0; i<in_height; i++) {
          for (int j=0; j<in_width; j++) {
            for (int k=0; k<in_channels; k++) {
              gradient_input->tensor<float, 4>()(b, i, j, k) = 0;
            }
          }
        }
      }
      // filter gradient
      for (int i=0; i<out_height; i++) {
        for (int j=0; j<out_width; j++) {
          for (int fi=0; fi<filter_height; fi++) {
            for (int fj=0; fj<filter_width; fj++) {
              int in_i = h_stride * i + fi;
              int in_j = w_stride * j + fj;
              for (int fk=0; fk<in_channels; fk++) {
                for (int k=0; k<out_channels; k++) {
                  double sum = 0.;
                  for (int b=0; b<batch_size; b++) {
                    sum +=
                      gradients.tensor<float, 4>()(b, i, j, k) *
                      input.tensor<float, 4>()(b, in_i, in_j, fk);
                    gradient_input->tensor<float, 4>()(b, in_i, in_j, k) +=
                      gradients.tensor<float, 4>()(b, i, j, k) *
                      filter.tensor<float, 6>()(i, j, fi, fj, fk, k);
                  }
                  gradient_filter->tensor<float, 6>()(i, j, fi, fj, fk, k) = sum;
                }
              }
            }
          }
        }
      }
    }
};

//TODO: get this working with half-precision
//REGISTER_KERNEL_BUILDER(Name("LocConn")
//    .Device(DEVICE_CPU)
//    .TypeConstraint<DT_HALF>("T"),
//    LocConnOp);
REGISTER_KERNEL_BUILDER(Name("LocConn")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    LocConnOp);
REGISTER_KERNEL_BUILDER(Name("LocConn")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    LocConnOp);

}

