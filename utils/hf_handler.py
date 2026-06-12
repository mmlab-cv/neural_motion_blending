from huggingface_hub import hf_hub_download
from huggingface_hub._snapshot_download import snapshot_download
import os

REPO_NAME = 'inbar2344/AnyTop'

def get_dependencies():
    dependencies_path = snapshot_download(repo_id=REPO_NAME)
    print('Data dependencies are cached at [{}]'.format(dependencies_path))
    link_all_checkpoints(dependencies_path)
    return dependencies_path

def link_all_checkpoints(dependencies_path):
    link_checkpoints(os.path.join(dependencies_path, 'checkpoints'), 'save')  # anytop checkpoints
    link_data(os.path.join(dependencies_path, 'dataset'), 'dataset')  # anytop dataset dependencies


def link_checkpoints(src_dir, dst_dir):
    assert os.path.isdir(src_dir)
    os.makedirs(dst_dir, exist_ok=True)
    all_subdirs = [subdir for subdir in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, subdir))]
    for subdir in all_subdirs:
        if not os.path.exists(os.path.join(dst_dir, subdir)):
            os.symlink(os.path.join(src_dir, subdir), os.path.join(dst_dir, subdir))

def get_all_files(directory):
    files = []
    for root, dirs, file_list in os.walk(directory):
        for file in file_list:
            relative_file_path = os.path.relpath(os.path.join(root, file), start=directory)
            files.append(relative_file_path)
    return files

def link_data(src_dir, dst_dir):
    assert os.path.isdir(src_dir)
    os.makedirs(dst_dir, exist_ok=True)
    all_files = get_all_files(src_dir)
    for f in all_files:
        file_path = os.path.join(dst_dir, f)
        if not os.path.exists(file_path):
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            os.symlink(os.path.join(src_dir, f), file_path)