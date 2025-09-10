from abc import ABC, abstractmethod

class AbstractDatabaseManager(ABC):
    @abstractmethod
    def create_table(self, table_name, **columns):
        """Tạo bảng với tên và cột"""
        pass

    @abstractmethod
    def add_entry(self, table, **kwargs):
        """Thêm dữ liệu vào bảng"""
        pass

    @abstractmethod
    def get_all(self, table):
        """Lấy tất cả dữ liệu"""
        pass

    @abstractmethod
    def get_by_id(self, table, entry_id):
        """Lấy dữ liệu theo ID"""
        pass

    @abstractmethod
    def update_entry(self, table, entry_id, **kwargs):
        """Cập nhật bản ghi theo ID"""
        pass

    @abstractmethod
    def delete_entry(self, table, entry_id):
        """Xóa bản ghi và cập nhật lại ID"""
        pass

    @abstractmethod
    def filter_entries(self, table, **filters):
        """Lọc dữ liệu theo điều kiện"""
        pass

    @abstractmethod
    def sort_entries(self, table, column, descending=False):
        """Sắp xếp dữ liệu theo cột"""
        pass

    @abstractmethod
    def close(self):
        """Đóng kết nối database"""
        pass
