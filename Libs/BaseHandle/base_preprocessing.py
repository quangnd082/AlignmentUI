from abc import ABC, abstractmethod


class HandleImage(ABC):
    @abstractmethod
    def find_blobs(self, *args, **kwargs):
        pass

    @abstractmethod
    def find_contours(self, *args, **kwargs):
        pass

    @abstractmethod
    def remove_blobs(self, *args, **kwargs):
        pass

    @abstractmethod
    def draw_contours(self, *args, **kwargs):
        pass

    @abstractmethod
    def find_circles(self, *args, **kwargs):
        pass

    @abstractmethod
    def draw_circles(self, *args, **kwargs):
        pass

    @abstractmethod
    def preprocess_image(self, *args, **kwargs):
        pass

    @abstractmethod
    def ndarray_to_qpixmap(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_canvas(self, *args, **kwargs):
        pass
