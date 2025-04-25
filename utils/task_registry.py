from typing import Any, Type
from models.base import BaseModel

class TaskRegistry():
    def __init__(self):
        self.model_classes = {}
        self.model_args = {}
        # self.train_cfgs = {}
    
    def register(self, name: str, model_class: Type[BaseModel], **kwargs):
        self.model_classes[name] = model_class
        self.model_args[name] = kwargs

    def get_task_names(self) -> list[str]:
        return list(self.model_classes.keys())

    def get_model_class(self, name: str) -> Type[BaseModel]:
        return self.model_classes[name]
    
    def get_model_args(self, name: str) -> dict[str, Any]:
        return self.model_args[name]