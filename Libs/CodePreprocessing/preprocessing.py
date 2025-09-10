import os
import shutil
import random
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def crop_image_percent(image_path, h_range=(3, 94), v_range=(45, 72)):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    start_x, end_x = int(w * h_range[0] / 100), int(w * h_range[1] / 100)
    start_y, end_y = int(h * v_range[0] / 100), int(h * v_range[1] / 100)
    
    start_x, end_x = max(0, start_x), min(w, end_x)
    start_y, end_y = max(0, start_y), min(h, end_y)
    
    cropped_img = image[start_y:end_y, start_x:end_x]
    return cropped_img

def resize_image(image_path, size_image=640):
    image = cv2.imread(image_path)
    image_resize = cv2.resize(image, (size_image, size_image))
    return image_resize

def crop_image_path(source_dir, resize_dir, crop_percent=False):
    os.makedirs(resize_dir, exist_ok=True)
    image_files = sorted(os.listdir(source_dir))
    for file in image_files:
        source_path = os.path.join(source_dir, file)
        if crop_percent == True:
            cropped_img = crop_image_percent(source_path)
        else:
            cropped_img = resize_image(source_path)
        output_path = os.path.join(resize_dir, file)
        cv2.imwrite(output_path, cropped_img)
        print(f"Ảnh đã được lưu tại {output_path}")
        
def split_dataset(source_dir, even_dir, odd_dir):
    os.makedirs(even_dir, exist_ok=True)
    os.makedirs(odd_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(source_dir))
    
    for i, file in enumerate(image_files):
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(even_dir if i % 2 == 0 else odd_dir, file)
        shutil.copy(src_path, dst_path)

def split_train_val_test(source_dir, **kwargs):
    train_dir = kwargs['train_dir']
    val_dir = kwargs['val_dir']
    test_dir = kwargs['test_dir']
    train_split = kwargs['train_split']
    val_split = kwargs['val_split']

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(source_dir))
    file_list = [f for f in image_files if os.path.isfile(os.path.join(source_dir, f))]
    quantity_files = len(file_list)
    
    quantity_files_train = int(quantity_files * train_split)
    quantity_files_val = int(quantity_files * val_split)
    
    random.shuffle(image_files)
    random.seed(42)
    
    for i, file in enumerate(image_files):
        if i - quantity_files_train < 0:
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(train_dir)
            shutil.copy(src_path, dst_path)
        elif i - quantity_files_train - quantity_files_val >= 0:
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(test_dir)
            shutil.copy(src_path, dst_path)
        else:
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(val_dir)
            shutil.copy(src_path, dst_path)

def show_random_images(folder_path, num_images=16):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    if len(image_files) == 0:
        print("Thư mục không chứa ảnh nào!")
        return
    if len(image_files) < num_images:
        print(f"Chỉ có {len(image_files)} ảnh trong thư mục, hiển thị tất cả.")
        num_images = len(image_files)
        
    selected_images = random.sample(image_files, num_images)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for ax, img_name in zip(axes.flatten(), selected_images):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def is_image(file_path):
    """Kiểm tra xem file có phải là ảnh không dựa vào phần mở rộng."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    return file_path.suffix.lower() in image_extensions

def copy_images_recursive(source_folder, destination_folder):
    """Duyệt qua toàn bộ cây thư mục và sao chép ảnh vào thư mục đích."""
    source_folder = Path(source_folder)
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)
    
    for root, _, files in os.walk(source_folder):
        for file in files:
            file_path = Path(root) / file
            if is_image(file_path):
                dest_path = destination_folder / file_path.name
                
                # Tránh ghi đè file trùng tên
                counter = 1
                while dest_path.exists():
                    dest_path = destination_folder / f"{file_path.stem}_{counter}{file_path.suffix}"
                    counter += 1
                
                shutil.copy2(file_path, dest_path)
                print(f"Đã sao chép: {file_path} -> {dest_path}")