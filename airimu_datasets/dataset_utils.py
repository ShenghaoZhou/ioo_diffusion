import torch


def imu_seq_collate(data):
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    init_pos = torch.stack([d['init_pos'] for d in data])
    init_rot = torch.stack([d['init_rot'] for d in data])
    init_vel = torch.stack([d['init_vel'] for d in data])

    dt = torch.stack([d['dt'] for d in data])

    return {
        'dt': dt,
        'acc': acc,
        'gyro': gyro,

        'gt_pos': gt_pos,
        'gt_vel': gt_vel,
        'gt_rot': gt_rot,

        'init_pos': init_pos,
        'init_vel': init_vel,
        'init_rot': init_rot,
    }


def load_data_with_padding(data, key):
    """
    Inspired by https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/14
    we pad the data to the same length for batch loading, 
    the model can ignore the padding by identifying padded zeros 
    """
    load_data = [d[key] for d in data]
    # pad_len = len(load_data[0]) - len(load_data[-1])
    # if pad_len > 0:
    #     load_data[-1] = torch.cat([load_data[-1],
    #                               torch.zeros(pad_len, load_data[-1].shape[1])])
    if key in ["init_pos", "init_vel", "init_rot"]:
        return torch.stack(load_data)
    # return torch.stack(load_data)
    batch = torch.nn.utils.rnn.pad_sequence(load_data, batch_first=True)
    return batch


def custom_collate(data):
    dt = torch.stack([d['dt'] for d in data])
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])
    rot = torch.stack([d['rot'] for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    init_pos = torch.stack([d['init_pos'] for d in data])
    init_rot = torch.stack([d['init_rot'] for d in data])
    init_vel = torch.stack([d['init_vel'] for d in data])

    return {'dt': dt, 'acc': acc, 'gyro': gyro, 'rot': rot, }, \
        {'pos': init_pos, 'vel': init_vel, 'rot': init_rot, }, \
        {'gt_pos': gt_pos, 'gt_vel': gt_vel, 'gt_rot': gt_rot, }


def collate_with_padding(data):
    keys = data[0].keys()
    # keys = ['dt', 'acc', 'gyro', 'rot', 'gt_pos', 'gt_vel',
    #         'gt_rot', 'init_pos', 'init_vel', 'init_rot', 'bias_g', 'bias_a']
    variables = {k: load_data_with_padding(data, k) for k in keys}
    return variables


def collate_with_padding_ori9(data):
    keys = ['dt', 'acc', 'gyro', 'rot', 'gt_pos', 'gt_vel',
            'gt_rot', 'init_pos', 'init_vel', 'init_rot']
    variables = {k: load_data_with_padding(data, k) for k in keys}
    rot = variables['init_rot'][:, 0:1, :]
    pad_acc = rot.Inv() * torch.tensor([0., 0., 9.81007],
                                       dtype=variables['dt'].dtype).repeat(len(data), 9, 1)
    pad_others = torch.zeros(len(data), 9, 3, dtype=variables['dt'].dtype)
    for key, val in variables.items():
        # idea: for acc, prepend gravity
        # for other data, prepend zeros
        # then the data remains the same size, so they can be loaded in batch
        # this should happen for each window
        # expected shape: (B, 209, 3)
        if key == "acc":
            variables[key] = torch.cat([pad_acc, val], dim=1)
        elif key == "gyro":
            variables[key] = torch.cat([pad_others, val], dim=1)

    return variables


def padding_collate(data, pad_len=1, use_gravity=True):
    B = len(data)
    input_data, init_state, label = custom_collate(data)

    if use_gravity:
        iden_acc_vector = torch.tensor(
            [0., 0., 9.81007], dtype=input_data['dt'].dtype).repeat(B, pad_len, 1)
    else:
        iden_acc_vector = torch.zeros(
            B, pad_len, 3, dtype=input_data['dt'].dtype)

    pad_acc = init_state['rot'].Inv() * iden_acc_vector
    pad_gyro = torch.zeros(B, pad_len, 3, dtype=input_data['dt'].dtype)

    input_data["acc"] = torch.cat([pad_acc, input_data['acc']], dim=1)
    input_data["gyro"] = torch.cat([pad_gyro, input_data['gyro']], dim=1)

    return input_data, init_state, label


collate_fcs = {
    "base": custom_collate,
    "padding": padding_collate,
    "padding9": lambda data: padding_collate(data, pad_len=9),
    "padding1": lambda data: padding_collate(data, pad_len=1),
    "Gpadding": lambda data: padding_collate(data, pad_len=9, use_gravity=False),
}
