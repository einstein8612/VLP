from utils.task_registry import TaskRegistry

from .rf import RF
from .mlp import MLP
from .mlp_online import MLPOnline
from .mlp_online_pico import MLPOnlinePico

from .pico_interface import PicoInterface

def register_models() -> TaskRegistry:
    """
    Register all models in the task registry.
    """
    task_registry = TaskRegistry()
    
    # Register models here
    task_registry.register("RF", RF)
    task_registry.register("RF-TINY", RF, n_estimators=6, max_depth=10)
    task_registry.register("MLP", MLP)
    task_registry.register("MLP-TINY", MLP, epochs=25)
    task_registry.register("MLP-TINY-NORMALISE", MLP, epochs=25, normalize=True)
    task_registry.register("MLP-ONLINE-TINY", MLPOnline, data_npy_path="./dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy", epochs=25)
    task_registry.register("MLP-ONLINE-PICO", MLPOnlinePico, data_npy_path="./dataset/heatmaps/heatmap_176_augmented_4_downsampled_4/augmented.npy", epochs=50)
    task_registry.register("PICO-INTERFACE", PicoInterface, serial_port='COM5')

    return task_registry