import json
import shutil
from BaseHandle.base_handle_file_json import HandleJSON
import os


class HandleJsonPBA(HandleJSON):
    def add(self, name_model, config: dict, local_path='Settings/ModelSettings'):
        shutil.copytree('Settings/ModelSettings/Default', f'{local_path}/{name_model}', dirs_exist_ok=True)
        self.save(name_model, config, local_path, overwrite=True)

    def delete(self, name_model, local_path='Settings/ModelSettings'):
        path = os.path.join(local_path, name_model)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"Đã xóa: {path}")
            except Exception as e:
                print(f"Lỗi khi xóa {path}: {e}")
        else:
            print(f"Không tìm thấy: {path}")

    def save(self, name_model, config: dict, local_path='Settings/ModelSettings', overwrite=False):
        path = f'{local_path}/{name_model}/config.json'

        # 1. Load config hiện tại (nếu có)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                current_config = json.load(f)
        except FileNotFoundError:
            current_config = {}

        if overwrite:
            # 2A. Chỉ giữ lại những gì có trong config mới → Ghi đè hoàn toàn
            new_config = config
        else:
            # 2B. Giữ lại key cũ, chỉ cập nhật/thêm key mới
            new_config = current_config.copy()
            for key, value in config.items():
                if key not in new_config or new_config[key] != value:
                    new_config[key] = value

        # 3. Ghi lại file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=4, ensure_ascii=False)

    def load(self, file_path='Settings/ModelSettings/Default'):
        config_path = os.path.join(file_path, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except json.JSONDecodeError:
                return None
        return None

    def update_config_key(self, name_model, key, value, local_path='Settings/ModelSettings'):
        path = f'{local_path}/{name_model}/config.json'

        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}

        data[key] = value  # Thêm hoặc cập nhật key

        with open(path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    name_model_1 = 'Trung'
    name_model_2 = 'Den'
    config = {
        'trung': 1,
        'den': 555
    }
    a = HandleJsonPBA()
    a.add(name_model_1, config)
    a.add(name_model_2, config)
