import os
import numpy as np
import torch
from ultralytics import YOLO

def test_yolov11_model(
    weights_path: str,
    data_config_path: str,
    custom_save_dir: str,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
):
    """
    使用指定权重和数据集配置，对测试集进行验证并打印/记录包括平均Precision、Recall、F1、mAP@0.5、mAP@0.5:0.95在内的指标，
    并将可视化推理结果(带检测框)保存到 custom_save_dir。

    注意:
    - Ultralytics v8.3.9 版本中，mp, mr, map50, map 等是 float 属性，而不是方法。
    - f1 是每个类别的列表，需要自己取 mean。
    """
    # 1. 加载模型
    model = YOLO(weights_path)

    # 2. 执行验证
    #    save=True 会输出推理后的可视化图片；save_dir= 指定保存目录
    results = model.val(
        data=data_config_path,
        device=device,
        split='test',     # 如果 data.yaml 中有 'test:' 字段
        batch=2,
        imgsz=640,
        save=True,
        save_dir=custom_save_dir  # 指定输出图片保存目录
    )

    # 3. 获取各项指标
    #    results.box 是一个 Metric 对象，里边的:
    #      - p, r, f1:  分别是每个类别的列表
    #      - mp, mr, map50, map: 这些是已计算好的 float（平均Precision、平均Recall、mAP@0.5、mAP@0.5:0.95）
    box_metrics = results.box

    mp = box_metrics.mp            # mean precision (float)
    mr = box_metrics.mr            # mean recall (float)
    f1_list = box_metrics.f1       # 每个类别的 F1 列表
    mf1 = float(np.mean(f1_list))  # 取所有类别的平均 F1
    mAP_50 = box_metrics.map50     # mAP@0.5
    mAP_5095 = box_metrics.map     # mAP@0.5:0.95

    # 打印到控制台
    print(f"Precision (mean): {mp:.4f}")
    print(f"Recall (mean): {mr:.4f}")
    print(f"F1 (mean): {mf1:.4f}")
    print(f"mAP@0.5 (AP50): {mAP_50:.4f}")
    print(f"mAP@0.5:0.95 (AP): {mAP_5095:.4f}")

    # 4. 将结果写入 custom_save_dir/test_results.txt
    os.makedirs(custom_save_dir, exist_ok=True)
    txt_file_path = os.path.join(custom_save_dir, "original_new_rebuild_test_results.txt")
    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write("YOLOv11 Test Results\n")
        f.write(f"Precision (mean): {mp:.4f}\n")
        f.write(f"Recall (mean): {mr:.4f}\n")
        f.write(f"F1 (mean): {mf1:.4f}\n")
        f.write(f"mAP@0.5 (AP50): {mAP_50:.4f}\n")
        f.write(f"mAP@0.5:0.95 (AP): {mAP_5095:.4f}\n")

if __name__ == '__main__':
    # 示例权重路径与数据配置路径，请根据自身情况修改
    #weights = r'C:\Under_water_rebar\水下钢筋\yolo11-seg\codelast20241025\code\runs\detect\train_v8_LIME_pure_data\weights\best.pt'   # 你的YOLOv11权重
    weights = r'C:\Under_water_rebar\水下钢筋\yolo11-seg\codelast20241025\code\runs\detect\train_v8_new_data_rebuild\weights\best.pt'
    data_yaml = r"C:\Under_water_rebar\水下钢筋\yolo11-seg\codelast20241025\code\datasets\new_data_rebuild\new_data_rebuild.yaml"      # 包含 test: 路径的 data.yaml
    custom_save_dir = r"C:\Under_water_rebar\水下钢筋\yolo11-seg\codelast20241025\code\test_result\LIME_pure_rebuild_test_result_4_22"       # 指定保存可视化图片及 test_results.txt 的目录

    test_yolov11_model(weights, data_yaml, custom_save_dir)
