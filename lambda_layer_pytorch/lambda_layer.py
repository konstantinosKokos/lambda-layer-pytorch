from torch.nn import Module, Linear
from torch import Tensor, stack
from opt_einsum import contract
from typing import Optional
from warnings import warn


class LambdaLayer(Module):
    def __init__(self, input_dim: int, context_dim: int, atn_dim: int, output_dim: int, num_heads: int):
        """
            Implements a Lambda Layer.

        :param input_dim: The dimensionality of the input tensor.
        :param context_dim: The dimensionality of the context tensor.
        :param atn_dim: The dimensionality of the query, key and value projections.
        :param output_dim: The dimensionality of the output.
        :param num_heads: The count of independent query projections.
        """
        if output_dim % num_heads != 0:
            warn('num_heads should be a divisor of output_dim')
        super(LambdaLayer, self).__init__()
        self.query_projection = Linear(input_dim, atn_dim * num_heads)
        self.key_projection = Linear(context_dim, atn_dim)
        self.value_projection = Linear(context_dim, output_dim//num_heads)
        self.num_heads = num_heads

    def compute_lambda(self, cs: Tensor, es: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """
            Computes the linear attention function to be applied to the query.

        :param cs: The context tensor :: (batch_size, ctx_len, ctx_dim)
        :param es: The relative positional embedding tensor :: (input_len, ctx_len, atn_dim)
        :param mask: The masking tensor, indexing elements to ignore with 0 :: (batch_size, ctx_len). Defaults to None.
        :return: A tuple of tensors (content_lambda, position_lambda), where
            content_lambda :: (batch_size, atn_dim, output_dim//num_heads)
            position_lambda :: (batch_size, input_len, atn_dim, output_dim//num_heads)
        """
        ks = self.key_projection(cs)
        if mask is not None:
            ks[mask == 0] = -1e10
        ks = ks.softmax(dim=1)
        vs = self.value_projection(cs)
        content_lambda = ks.transpose(-2, -1)@vs
        position_lambda = contract('nmk,bmv->bnkv', es, vs)
        return content_lambda, position_lambda

    def forward(self, xs: Tensor, cs: Tensor, es: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
            Computes the attention-adjusted values for some input, context, relative embedding and mask.

        :param xs: The input tensor :: (batch_size, input_len, input_dim)
        :param cs: The context tensor :: (batch_size, ctx_len, ctx_dim)
        :param es: The relative position embedding tensor :: (input_len, ctx_len, atn_dim)
        :param mask: The masking tensor, indexing elements to ignore with 0 :: (batch_size, ctx_len). Defaults to None.
        :return:
            A new tensor of shape (batch_size, input_len, output_dim)
        """
        qs = stack(self.query_projection(xs).chunk(self.num_heads, -1), 1)
        c_lambda, p_lambda = self.compute_lambda(cs, es, mask)
        c_out = contract('bhnk,bkv->bnhv', qs, c_lambda)
        p_out = contract('bhnk,bnkv->bnhv', qs, p_lambda)
        return (p_out + c_out).flatten(-2)


def _test_layer():
    import torch
    from random import randint

    batch_size, input_len, ctx_len, input_dim, ctx_dim, atn_dim, output_dim = [randint(1, 300) for _ in range(7)]

    num_heads = randint(1, 5)
    output_dim = output_dim//num_heads * num_heads
    xs = torch.rand(batch_size, input_len, input_dim)
    ctx = torch.rand(batch_size, ctx_len, ctx_dim)
    embedding = torch.rand(input_len, ctx_len, atn_dim)

    mask = torch.randint(low=0, high=2, size=(batch_size, ctx_len))
    layer = LambdaLayer(input_dim, ctx_dim, atn_dim, output_dim, num_heads)
    output = layer(xs, ctx, embedding, mask)
    assert output.shape == (batch_size, input_len, output_dim)