import torch


def extract_input(dataset, slice_range, device="cpu", gtinit=False):

    acc = torch.stack([dataset[i]["acc"] for i in slice_range])
    gyro = torch.stack([dataset[i]["gyro"] for i in slice_range])
    dt = torch.stack([dataset[i]["dt"] for i in slice_range])
    init_pos = torch.stack([dataset[i]["init_pos"]
                           for i in slice_range])
    init_rot = torch.stack([dataset[i]["init_rot"]
                           for i in slice_range])
    init_vel = torch.stack([dataset[i]["init_vel"]
                            for i in slice_range])

    dt = dt.to(device)
    gyro = gyro.to(device)
    acc = acc.to(device)
    init_pos = init_pos.to(device)
    init_vel = init_vel.to(device)
    init_rot = init_rot.to(device)

    # FIXME: how to load everything here
    if gtinit:
        # FIXME: shape is not correct
        init_state = {
            "pos": init_pos[:, :1, :],
            "vel": init_vel[:, :1, :],
            "rot": init_rot[:, :1, :],
        }
    else:
        init_state = None
    return init_state, dt, gyro, acc,


def batch_integrate(integrator, dataset, init, device="cpu", gtinit=False, use_gt_rot=False):
    # FIXME: init seems to be useless
    integrator = integrator.to(device)
    integrator.eval()

    slice_range = range(len(dataset) - 1)
    init_state, dt, gyro, acc = extract_input(
        dataset, slice_range, device=device, gtinit=gtinit)

    state_first_batches = integrator(
        init_state=init_state, dt=dt,
        gyro=gyro, acc=acc,
        rot=init_state["rot"] if (
            use_gt_rot and init_state is not None) else None
    )

    # TODO: take care of the last batch, which might not be full batch size
    slice_range = range(len(dataset) - 1, len(dataset))
    init_state, dt, gyro, acc = extract_input(
        dataset, slice_range, device=device, gtinit=gtinit)

    state_last_batch = integrator(
        init_state=init_state, dt=dt,
        gyro=gyro, acc=acc,
        rot=init_state["rot"] if use_gt_rot else None
    )

    out_state = dict()
    for key in ["rot", "vel", "pos"]:
        out_state[key] = torch.cat([state_first_batches[key][..., -1:, :].cpu(),
                                    state_last_batch[key][..., -1:, :].cpu()
                                    ], dim=0)
    # rename rot
    out_state["orientations"] = out_state["rot"]

    # compute raw distance, for metric computation
    gt_pos = torch.stack([dataset[i]["gt_pos"][..., -1:, :]
                         for i in range(len(dataset))])
    gt_rot = torch.stack([dataset[i]["gt_rot"][..., -1:, :]
                         for i in range(len(dataset))])
    gt_vel = torch.stack([dataset[i]["gt_vel"][..., -1:, :]
                         for i in range(len(dataset))])

    out_state["pos_dist"] = (
        out_state["pos"][:, 0:, :] - gt_pos[:, 0:, :]).norm(dim=-1).T
    out_state["vel_dist"] = (
        out_state["vel"][:, 0:, :] - gt_vel[:, 0:, :]).norm(dim=-1).T
    out_state["rot_dist"] = (
        (gt_rot[:, 0:, :].Inv() @ out_state["orientations"][:, 0:, :]).Log()).norm(dim=-1).T
    return out_state
