import os

import torch
import yaml
from ultralytics import YOLO  # 导入YOLO模型
from QtFusion.path import abs_path
device = "0" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':  
    workers = 1
    batch = 32
    data_name = ""
    data_path = abs_path(f'datasets/{data_name}/{data_name}.yaml', path_type='current')  # 数据集的yaml的绝对路径
    unix_style_path = data_path.replace(os.sep, '/')

    # 获取目录路径
    directory_path = os.path.dirname(unix_style_path)
    # 读取YAML文件，保持原有顺序
    with open(data_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)




    # 修改path项
    if 'path' in data:
        data['path'] = directory_path
        # 将修改后的数据写回YAML文件
        with open(data_path, 'w') as file:
            yaml.safe_dump(data, file, sort_keys=False)
    if 'train' in data and 'val' in data and 'test' in data:
        data['train'] = directory_path + '/train'
        data['val'] = directory_path + '/val'
        data['test'] = directory_path + '/test'
        # 将修改后的数据写回YAML文件
        with open(data_path, 'w') as file:
            yaml.safe_dump(data, file, sort_keys=False)

    model = YOLO(model='./ultralytics/cfg/models/v11/yolo11.yaml', task='detect').load('./yolo11s.pt')  # 加载预训练的YOLOv11模型

    results2 = model.train(  # 开始训练模型
        data=data_path,  # 指定训练数据的配置文件路径
        device=device,  # 自动选择进行训练
        workers=workers,  # 指定使用2个工作进程加载数据
        imgsz=640,  # 指定输入图像的大小为640x640
        epochs=120
        ,  # 指定训练epoch
        batch=batch,  # 指定每个批次的大小为8
        name='train_v11' + data_name  # 指定训练任务的名称
    )
