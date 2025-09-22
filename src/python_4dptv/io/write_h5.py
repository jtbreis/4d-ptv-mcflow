import h5py


def write_h5(filename, data, metadata):
    with h5py.File(filename, 'w') as f:
        for key, meta_values in metadata.items():
            f.attrs[key] = meta_values
        for frame, values in data.items():
            grp = f.create_group(f'frame{int(frame):05d}')
            for key, value in values.items():
                grp.create_dataset(key, data=value)


def write_h5_multiple_camera(filename, data, metadata):
    with h5py.File(filename, 'w') as f:
        for key, meta_values in metadata.items():
            f.attrs[key] = meta_values
        for cam_idx, cam_values in data.items():
            camgrp = f.create_group(cam_idx)
            for frame, values in cam_values.items():
                grp = camgrp.create_group(f'frame{int(frame):05d}')
                for key, value in values.items():
                    grp.create_dataset(key, data=value)
