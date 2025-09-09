import aihwkit
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import (
    WeightRemapType,
    WeightModifierType,
    WeightClipType,
)
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.simulator.presets.utils import IOParameters


def gen_rpu_config():
    rpu_config = InferenceRPUConfig()
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    rpu_config.mapping.max_input_size = 256
    rpu_config.mapping.max_output_size = 255
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC
    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu_config.clip.sigma = 2.0
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
    rpu_config.modifier.rel_to_actual_wmax = True
    rpu_config.modifier.std_dev = 0.08
    rpu_config.forward = IOParameters()
    rpu_config.forward.out_noise = 0.06
    rpu_config.forward.inp_res = 1 / (2**8 - 2)  # 8-bit resolution.
    rpu_config.forward.out_res = 1 / (2**8 - 2)  # 8-bit resolution.
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config

