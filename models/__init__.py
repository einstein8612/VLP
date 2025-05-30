from utils.task_registry import TaskRegistry

from .rf import RF
from .mlp import MLP
from .mlp_online import MLPOnline

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
    
    return task_registry