from utils.task_registry import TaskRegistry

from .rf import RF

def register_models() -> TaskRegistry:
    """
    Register all models in the task registry.
    """
    task_registry = TaskRegistry()
    
    # Register models here
    task_registry.register("RF", RF)
    task_registry.register("RF-TINY", RF, n_estimators=6, max_depth=10)
    
    return task_registry