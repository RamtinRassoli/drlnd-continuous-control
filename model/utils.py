import ruamel.yaml as yaml


def get_config(config_file):
    stream = open(config_file, "r")
    # yaml = yaml(typ='safe')  # default, if not specfied, is 'rt' (round-trip)
    config = yaml.safe_load(stream)

    return config


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)