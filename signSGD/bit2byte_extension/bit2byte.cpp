#include <iostream>
#include <torch/extension.h>

torch::Tensor compress(torch::Tensor src_tensor)
{
    //src_tensor is dim(8*-1) UInt8Tensor
    src_tensor[0].__irshift__(1);
    for (int i = 1; i < 8; i++)
    {
        src_tensor[0].__ilshift__(1);
        src_tensor[i].__irshift__(1);
        src_tensor[0].__ior__(src_tensor[i]);
    }

    return {src_tensor[0]};
}

torch::Tensor uncompress(torch::Tensor compressed_tensor, torch::Tensor dst_tensor)
{
    //compressed_tensor is dim(1*-1) UInt8Tensor
    //dst_tensor is dim(8*-1) UInt8Tensor (ones)

    for (int i = 7; i >= 0; i--)
    {
        dst_tensor[i].__iand__(compressed_tensor);
        compressed_tensor.__irshift__(1);
        dst_tensor[i].__ilshift__(1);
    }

    return {dst_tensor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compress", &compress, "compress");
    m.def("uncompress", &uncompress, "uncompress");
}