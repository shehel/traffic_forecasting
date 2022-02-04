from abc import ABC, abstractmethod


class DataTransform(ABC):

    @abstractmethod
    def pre_transform():
        pass
    
    @abstractmethod
    def post_transform():
        pass
