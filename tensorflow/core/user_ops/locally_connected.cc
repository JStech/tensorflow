#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

REGISTER_OP("LocConn")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {half, float, double} = DT_FLOAT")
    //.SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
Computes a 2-D locally connected layer given 4-D `input` and 6-D `filter` tensors.

(((TODO: write this)))
Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter tensor of shape `[out_height, out_width, filter_height,
filter_width, in_channels, out_channels]`, this op performs the following:

1. Flattens the filter to a 4-D matrix with shape
   `[out_height, out_width, filter_height * filter_width * in_channels,
        output_channels]`.
2. Evenly 
2. Extracts image patches from the input tensor to form a *virtual*
   tensor of shape `[batch, out_height, out_width,
   filter_height * filter_width * in_channels]`.
3. For each patch, right-multiplies the respective filter matrix and the image
   patch vector.

In detail, with the default NHWC format,

    output[b, i, j, k] =
        sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                        filter[di, dj, q, k]
)doc");

class LocConnOp : public OpKernel {
  public:
    explicit LocConnOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      const Tensor& input = context->input(0);
      const Tensor& filter = context->input(1);

      // check dimensions
      OP_REQUIRES(context, input.dims() == 4,
          errors::InvalidArgument("input must be 4-dimensional",
            input.shape().DebugString()));

      OP_REQUIRES(context, filter.dims() == 6,
          errors::InvalidArgument("filter must be 6-dimensional",
            filter.shape().DebugString()));

      const int64 in_height = input.dim_size(1);
      const int64 in_width = input.dim_size(2);
      const int64 in_depth = input.dim_size(3);
      OP_REQUIRES(
          context, in_depth == filter.dim_size(4),
          errors::InvalidArgument("input and filter must have the same depth: ",
            in_depth, " vs ", filter.dim_size(4)));

      // output dimensions
      const int64 batch = input.dim_size(0);
      const int out_height = static_cast<int>(filter.dim_size(0));
      const int out_width = static_cast<int>(filter.dim_size(1));
      const int out_depth = static_cast<int>(filter.dim_size(5));

      // filter dimensions
      const int filter_height = static_cast<int>(filter.dim_size(2));
      const int filter_width = static_cast<int>(filter.dim_size(3));

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

      int h_stride = (in_height - filter_height + 1)/out_height;
      int w_stride = (in_width - filter_width + 1)/out_width;

      for (int b=0; b<batch; b++) {
        for (int k=0; k<out_depth; k++) {
          for (int i=0; i<out_height; i++) {
            for (int j=0; j<out_width; j++) {
              double sum = 0.;
              for (int fi=0; fi<filter_height; fi++) {
                for (int fj=0; fj<filter_width; fj++) {
                  for (int fk=0; fk<in_depth; fk++) {
                    sum += 
                  }
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
