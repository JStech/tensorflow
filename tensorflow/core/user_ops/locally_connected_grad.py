from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient("LocConn")
def _loc_conn_grad(op, grad):
  """The gradients for `loc_conn`.

  Args:
    op: The `loc_conn` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `loc_conn` op.

  Returns:
    Gradients with respect to the input of `loc_conn`.
  """
  print(array_ops.shape(grad))
  print(len(op.inputs))
  return None
