from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from BaseHandle.base_connect_database import AbstractDatabaseManager
import pytz
VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")


class DatabaseManager(AbstractDatabaseManager):
    def __init__(self, db_name="database.db"):
        """Khởi tạo database"""
        self.engine = create_engine(f"sqlite:///{db_name}", echo=False)
        self.Base = declarative_base()
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        self.tables = {}  # Lưu bảng đã tạo

    def create_table(self, table_name, **columns):
        """
        Tạo bảng với tên bảng và danh sách cột tùy chỉnh.
        Ví dụ: create_table("users", name=String, age=Integer)
        """
        if table_name in self.tables:
            return self.tables[table_name]  # Trả về bảng nếu đã tạo trước đó

        def vn_now():
            return datetime.now(VN_TZ)
        
        # Tạo class động cho bảng
        attrs = {
            "__tablename__": table_name,
            "id": Column(Integer, primary_key=True, autoincrement=True)
        }
        for col_name, col_type in columns.items():
            attrs[col_name] = Column(col_type, nullable=False)
            
        attrs["Time"] = Column(DateTime, default=vn_now)
        
        table_class = type(table_name, (self.Base,), attrs)
        self.tables[table_name] = table_class  # Lưu bảng vào dictionary
        self.Base.metadata.create_all(self.engine)
        return table_class  # Trả về class bảng

    def add_entry(self, table, **kwargs):
        """Thêm dữ liệu vào bảng"""
        new_entry = table(**kwargs)
        self.session.add(new_entry)
        self.session.commit()
        return new_entry

    def get_all(self, table):
        """Lấy danh sách tất cả dữ liệu trong bảng"""
        return self.session.query(table).all()

    def get_by_id(self, table, entry_id):
        """Tìm dữ liệu theo ID"""
        return self.session.query(table).filter_by(id=entry_id).first()

    def update_entry(self, table, entry_id, **kwargs):
        """Cập nhật dữ liệu theo ID"""
        entry = self.get_by_id(table, entry_id)
        if entry:
            for key, value in kwargs.items():
                setattr(entry, key, value)
            self.session.commit()
            return entry
        return None

    def delete_entry(self, table, entry_id):
        """Xóa dữ liệu theo ID và cập nhật lại ID cho các dòng còn lại"""
        entry = self.get_by_id(table, entry_id)
        if entry:
            self.session.delete(entry)
            self.session.commit()

            # Cập nhật lại ID của các dòng còn lại
            all_entries = self.get_all(table)
            for index, row in enumerate(all_entries, start=1):
                row.id = index
            self.session.commit()

            return True
        return False

    def filter_entries(self, table, **filters):
        """Lọc dữ liệu theo điều kiện"""
        query = self.session.query(table)
        for key, value in filters.items():
            query = query.filter(getattr(table, key) == value)
        return query.all()

    def sort_entries(self, table, column, descending=False):
        """Sắp xếp dữ liệu theo cột"""
        query = self.session.query(table)
        order_by_column = getattr(table, column)
        return query.order_by(order_by_column.desc() if descending else order_by_column.asc()).all()

    def close(self):
        """Đóng session"""
        self.session.close()

if __name__ == '__main__':
    # === CÁCH SỬ DỤNG ===
    db = DatabaseManager(db_name="res/Database/my_database.db")  # Tạo database

    # Tạo bảng "users" với cột: name (String), age (Integer), email (String)
    UserTable = db.create_table("users", name=String, age=Integer, email=String)

    # Thêm dữ liệu
    db.add_entry(UserTable, name="Alice", age=25, email="alice@example.com")
    db.add_entry(UserTable, name="Bob", age=30, email="bob@example.com")
    db.add_entry(UserTable, name="Charlie", age=22, email="charlie@example.com")

    # Lấy danh sách users
    print("Danh sách users:", db.get_all(UserTable))

    # Xóa user có ID=2 (Bob) và cập nhật lại ID
    db.delete_entry(UserTable, 2)

    # Lấy danh sách users sau khi xóa
    print("Danh sách users sau khi xóa:", db.get_all(UserTable))

    # Đóng kết nối
    db.close()
