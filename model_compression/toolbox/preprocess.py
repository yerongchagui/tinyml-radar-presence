import numpy as np

def preprocess(data_in, chirp_count=16, num_samples=64, normalization=False):
    # normalization = True
    if normalization:
        data_in = (data_in - data_in.min(axis=(1, 2, 3), keepdims=True)) / (data_in.max(axis=(1, 2, 3), keepdims=True))
    data_out = []
    data_out_slow = []
    data_list = []
    for i_frame, data in enumerate(data_in):
        if i_frame < int(chirp_count):
            data_list.append(np.copy(data))  # divide by data.max()
        else:
            del data_list[0]
            data_list.append(np.copy(data.astype(np.float32)))
            # creating slow time data frame
            data_slow = np.stack(data_list, axis=0)
            data_slow = np.transpose(data_slow.mean(axis=2), [1, 0, 2])

            # remove means for MTI
            data = data - data.mean(axis=1, keepdims=True)
            data_slow = data_slow - data_slow.mean(axis=1, keepdims=True)

            # Do Range-Doppler processing
            # windowing
            data = data * (np.hamming(num_samples) * (np.hamming(chirp_count)[np.newaxis]).T[np.newaxis])
            data_slow = data_slow * (np.hamming(num_samples) * (np.hamming(chirp_count)[np.newaxis]).T[np.newaxis])

            # FFTs
            data = np.fft.fft2(data, axes=[2, 1])[..., :int(num_samples // 2)]
            data_slow = np.fft.fft2(data_slow, axes=[2, 1])[..., :int(num_samples // 2)]
            # FFT shifts
            data = np.fft.fftshift(data, axes=1)
            data_slow = np.fft.fftshift(data_slow, axes=1)
            data_out.append(data)
            data_out_slow.append(data_slow)

    # data_out[0].shape: (3, 16, 32)

    data = np.stack([np.real(data_out), np.imag(data_out), np.real(data_out_slow), np.imag(data_out_slow)], axis=-1)
    # data.shape:  (1185, 3, 16, 32, 4)
    macro = data[..., 0:2]
    micro = data[..., 2:4]
    macro = np.moveaxis(macro, 1, -2)
    # macro.shape: (1185, 16, 32, 3, 2)
    macro = macro.reshape((macro.shape[0], macro.shape[1], macro.shape[2], -1))
    micro = np.moveaxis(micro, 1, -2)
    # micro.shape: (1185, 16, 32, 3, 2)
    micro = micro.reshape((micro.shape[0], micro.shape[1], micro.shape[2], -1))
    # macro (1185, 16, 32, 6)
    # micro (1185, 16, 32, 6)
    data = np.concatenate([macro, micro], axis=-1)  # [16, 32, 12]

    # antenna 1: idx: [..., 0,1,6,7]
    # antenna 2: idx: [..., 2, 3, 8, 9]
    # antenna 3: idx: [..., 4, 5, 10, 11]

    """
    data = np.stack([np.abs(data_out), np.abs(data_out_slow)], axis=-1)
    data = data.reshape((-1, data.shape[-3], data.shape[-2], data.shape[-1]))
    data = data[:,:,:16,:]
    data = data/np.amax(data, axis=(1,2), keepdims=True)
    data[data<0.05]=0
    """
    data = data[:, :, :10, ...]
    if data.shape[-1] > 8:
        #return data[..., [0,1,4,5,6,7,10,11]].astype(np.float16)
        return data[..., [0,1,4,5,6,7,10,11]].astype(np.float32)

    #return data.astype(np.float16)

    return data.astype(np.float32)
