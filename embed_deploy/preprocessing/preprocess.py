import numpy as np

def preprocess_data(data, chirp_count=16, num_samples=64, normalisation=False):
    data_out = []
    data_out_slow = []
    data_list = []

    for i_frame, data in enumerate(data):
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

    data = np.stack([np.real(data_out), np.imag(data_out), np.real(data_out_slow), np.imag(data_out_slow)], axis=-1)
    macro = data[..., 0:2]
    micro = data[..., 2:4]
    macro = np.moveaxis(macro, 1, -2)
    macro = macro.reshape((macro.shape[0], macro.shape[1], macro.shape[2], -1))
    micro = np.moveaxis(micro, 1, -2)
    micro = micro.reshape((micro.shape[0], micro.shape[1], micro.shape[2], -1))
    data = np.concatenate([macro, micro], axis=-1)  # [16, 32, 12]

    data = data[:, :, :10, ...]
    if data.shape[-1] > 8:
        return data[..., [0,1,4,5,6,7,10,11]].astype(np.float32)

    return data.astype(np.float32)


if __name__ == "__main__":
    raw_sample = np.load("embed_deploy/train_samples/LONG2022_10_24_15_37_48radar.npy")
    target_processed_sample = np.load("embed_deploy/train_samples/PROCESSED_LONG2022_10_24_15_37_48radar.npy")
    print(raw_sample.shape)
    print(target_processed_sample.shape)

    processed_sample = preprocess_data(raw_sample / 4095.0)

    assert np.allclose(processed_sample, target_processed_sample)