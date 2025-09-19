import h5py


def write_h5(filename, data, metadata):
    with h5py.File(filename, 'w') as f:
        for key, meta_values in metadata.items():
            f.attrs[key] = meta_values
        for frame, values in data.items():
            grp = f.create_group(f'frame{int(frame):05d}')
            for key, value in values.items():
                grp.create_dataset(key, data=value)
