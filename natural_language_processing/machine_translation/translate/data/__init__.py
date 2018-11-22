def pad_sequence(sequences, batch_first=False, padding_value=0,
                 padding_first=False):
    """ A modification of the pytorch implementation so that the padding can
    be added before the sequence, instead of after.
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            if padding_first:
                out_tensor[i, -length:, ...] = tensor
            else:
                out_tensor[i, :length, ...] = tensor
        else:
            if padding_first:
                out_tensor[-length:, i, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor
    return out_tensor
