import nnvm
import nnvm.compiler

import tvm
import onnx
import numpy as np
import scipy.misc

from PIL import Image
from tvm.contrib import graph_runtime


def download(url, path, overwrite=False):
    import os
    if os.path.isfile(path) and not overwrite:
        print('File {} existed, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
    except:
        import urllib
        urllib.urlretrieve(url, path)


def main():
    model_url = ''.join(['https://gist.github.com/zhreshold/',
                         'bcda4716699ac97ea44f791c24310193/raw/',
                         '93672b029103648953c4e5ad3ac3aadf346a4cdc/',
                         'super_resolution_0.2.onnx'])
    download(model_url, 'super_resolution.onnx', True)
    # now you have super_resolution.onnx on disk
    onnx_model = onnx.load('super_resolution.onnx')
    # we can load the graph as NNVM compatible model
    sym, params = nnvm.frontend.from_onnx(onnx_model) 

    img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
    download(img_url, 'cat.png')
    img = Image.open('cat.png').resize((224, 224))
    img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
    img_y, img_cb, img_cr = img_ycbcr.split()
    x = np.array(img_y)[np.newaxis, np.newaxis, :, :]

    target = 'metal'
    # assume first input name is data
    input_name = sym.list_input_names()[0]
    shape_dict = {input_name: x.shape}
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

    ctx = tvm.metal(0)
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input(input_name, tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    output_shape = (1, 1, 672, 672)
    tvm_output = m.get_output(0, tvm.nd.empty(output_shape, dtype)).asnumpy()

    out_y = Image.fromarray(np.uint8((tvm_output[0, 0]).clip(0, 255)), mode='L')
    out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
    out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
    result = Image.merge('YCbCr', [out_y, out_cb, out_cr]).convert('RGB')
    # canvas = np.full((672, 672*2, 3), 255)
    # canvas[0:224, 0:224, :] = np.asarray(img)
    # canvas[:, 672:, :] = np.asarray(result)
    scipy.misc.imsave('./input.jpg', img)
    scipy.misc.imsave('./result.jpg', result)


if __name__ == '__main__':
    main()
