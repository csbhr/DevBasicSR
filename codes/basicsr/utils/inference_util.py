import torch


def inference_flipx4(model, input_tensor, reduce_type='mean'):
    """Inference with self-ensemble x4:
        normal, flip W, flip H, flip both H and W

    Args:
        model (nn.Module): The network model.
        input_tensor (Tensor): The input tensor with shape [..., H, W].
        reduce_type (Str): The reducing type when ensemble outputs,
            optional: 'mean' | 'median' | 'none', default: 'mean'.

    Returns:
        Tensor: The output tensor.
    """
    with torch.no_grad():
        output_list = []
        # normal
        output_f = model(input_tensor)
        if isinstance(output_f, list):
            output_f = output_f[-1]
        output_list.append(output_f)
        # flip W
        output_f = model(torch.flip(input_tensor, (-1,)))
        if isinstance(output_f, list):
            output_f = output_f[-1]
        output_f = torch.flip(output_f, (-1,))
        output_list.append(output_f)
        # flip H
        output_f = model(torch.flip(input_tensor, (-2,)))
        if isinstance(output_f, list):
            output_f = output_f[-1]
        output_f = torch.flip(output_f, (-2,))
        output_list.append(output_f)
        # flip both H and W
        output_f = model(torch.flip(input_tensor, (-2, -1)))
        if isinstance(output_f, list):
            output_f = output_f[-1]
        output_f = torch.flip(output_f, (-2, -1))
        output_list.append(output_f)
        # ensemble
        output_stack = torch.stack(output_list, dim=0)
        if reduce_type == 'mean':
            return torch.mean(output_stack, dim=0)
        elif reduce_type == 'median':
            return torch.median(output_stack, dim=0)[0]
        elif reduce_type == 'none':
            return output_stack
        else:
            raise NotImplementedError


def inference_flipx8(model, input_tensor, reduce_type='mean'):
    """Inference with self-ensemble x8:
        normal, flip W, flip H, flip both H and W
        Transpose HW and then: (normal, flip W, flip H, flip both H and W)

    Args:
        model (nn.Module): The network model.
        input_tensor (Tensor): The input tensor with shape [..., H, W].
        reduce_type (Str): The reducing type when ensemble outputs,
            optional: 'mean' | 'median' | 'none', default: 'mean'.

    Returns:
        Tensor: The output tensor.
    """
    output_flipx4 = inference_flipx4(model, input_tensor, reduce_type='none')

    output_flipx4_tran = inference_flipx4(model, torch.transpose(input_tensor, -1, -2), reduce_type='none')
    output_flipx4_tran = torch.transpose(output_flipx4_tran, -1, -2)

    # ensemble
    output_stack = torch.cat([output_flipx4, output_flipx4_tran], dim=0)
    if reduce_type == 'mean':
        return torch.mean(output_stack, dim=0)
    elif reduce_type == 'median':
        return torch.median(output_stack, dim=0)[0]
    elif reduce_type == 'none':
        return output_stack
    else:
        raise NotImplementedError


def inference_flipx16(model, input_tensor, reduce_type='mean'):
    """Inference with self-ensemble x16:
        Only support the input tensor with shape [B, T, C, H, W].
        normal, flip W, flip H, flip both H and W
        Transpose HW and then: (normal, flip W, flip H, flip both H and W)
        Flip T and then: (normal, flip W, flip H, flip both H and W)
        Flip T and then: (Transpose HW and then: (normal, flip W, flip H, flip both H and W))

    Args:
        model (nn.Module): The network model.
        input_tensor (Tensor): The input tensor with shape [B, T, C, H, W].
        reduce_type (Str): The reducing type when ensemble outputs,
            optional: 'mean' | 'median' | 'none', default: 'mean'.

    Returns:
        Tensor: The output tensor.
    """
    assert input_tensor.ndim == 5

    output_flipx8 = inference_flipx8(model, input_tensor, reduce_type='none')

    output_flipx8_flipT = inference_flipx8(model, torch.flip(input_tensor, (1,)), reduce_type='none')
    output_flipx8_flipT = torch.flip(output_flipx8_flipT, (2,))  # an additional dim due to stack operation

    # ensemble
    output_stack = torch.cat([output_flipx8, output_flipx8_flipT], dim=0)
    if reduce_type == 'mean':
        return torch.mean(output_stack, dim=0)
    elif reduce_type == 'median':
        return torch.median(output_stack, dim=0)[0]
    elif reduce_type == 'none':
        return output_stack
    else:
        raise NotImplementedError


if __name__ == '__main__':

    def func(x):
        return x

    a = torch.randn((2, 7, 32, 256, 512)).cuda()
    x4 = inference_flipx4(func, a, reduce_type='mean')
    diff_x4 = torch.sum(torch.abs(a - x4))
    x8 = inference_flipx8(func, a, reduce_type='mean')
    diff_x8 = torch.sum(torch.abs(a - x8))
    x16 = inference_flipx16(func, a, reduce_type='mean')
    diff_x16 = torch.sum(torch.abs(a - x16))
    print()
