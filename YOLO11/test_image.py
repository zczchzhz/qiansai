import random
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from hashlib import md5
from model import Web_Detector
from chinese_name_list import Label_list


def letterbox(image, target_size=640, color=(114, 114, 114)):
    """将图像resize并padding为正方形，保持宽高比"""
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 计算padding位置
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2

    # 创建画布并填充
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, (pad_top, pad_left)


def generate_color_based_on_name(name):
    # 使用哈希函数生成稳定的颜色
    hash_object = md5(name.encode())
    hex_color = hash_object.hexdigest()[:6]  # 取前6位16进制数
    # 修正括号闭合问题，并添加进制参数
    r = int(hex_color[0:2], 16)  # 添加16进制参数
    g = int(hex_color[2:4], 16)  # 添加16进制参数
    b = int(hex_color[4:6], 16)  # 添加16进制参数
    return (b, g, r)  # OpenCV 使用BGR格式


def calculate_polygon_area(points):
    return cv2.contourArea(points.astype(np.float32))


def draw_with_chinese(image, text, position, font_size=20, color=(255, 0, 0)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("simsun.ttc", font_size, encoding="unic")
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


def adjust_parameter(image_size, base_size=1000):
    max_size = max(image_size)
    return max_size / base_size


def draw_detections(image, info, alpha=0.2):
    name, bbox, conf, cls_id, mask = info['class_name'], info['bbox'], info['score'], info['class_id'], info['mask']
    adjust_param = adjust_parameter(image.shape[:2])
    spacing = int(20 * adjust_param)

    if mask is None:
        x1, y1, x2, y2 = map(int, bbox)
        aim_frame_area = (x2 - x1) * (y2 - y1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), int(3 * adjust_param))
        # 在类别名称后添加置信度
        display_text = f"{name} {conf:.2f}"  # 显示类别和置信度
        image = draw_with_chinese(image, display_text, (x1, y1 - int(30 * adjust_param)),
                                  font_size=int(35 * adjust_param))
    else:
        mask_points = np.concatenate(mask).astype(np.int32)
        aim_frame_area = calculate_polygon_area(mask_points)
        mask_color = generate_color_based_on_name(name)
        try:
            overlay = image.copy()
            cv2.fillPoly(overlay, [mask_points], mask_color)
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
            cv2.drawContours(image, [mask_points], -1, (0, 0, 255), int(8 * adjust_param))

            # 计算各类指标
            area = cv2.contourArea(mask_points)
            perimeter = cv2.arcLength(mask_points, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # 绘制文字信息（添加置信度）
            x, y = np.min(mask_points, axis=0)
            image = draw_with_chinese(image, f"{name} {conf:.2f}",  # 类别+置信度
                                      (x, y - int(30 * adjust_param)),
                                      font_size=int(35 * adjust_param))

            # 添加度量信息（包含置信度）
            metrics = [
                f"Area: {area:.1f}",
                f"Perimeter: {perimeter:.1f}",
                f"Circularity: {circularity:.2f}",
                f"Confidence: {conf:.2f}"  # 新增置信度行
            ]
            for i, text in enumerate(metrics):
                image = draw_with_chinese(image, text,
                                          (x, y - int(50 * adjust_param) - i * int(40 * adjust_param)),
                                          font_size=int(30 * adjust_param))
        except Exception as e:
            print(f"绘制错误: {e}")
    return image, aim_frame_area


def process_frame(model, image):
    # 保留原始图像副本
    original_image = image.copy()
    h, w = image.shape[:2]

    # 预处理尺寸
    scale = 1.0
    pad_top = pad_left = 0
    if h != 640 or w != 640:
        image, scale, (pad_top, pad_left) = letterbox(image)

    # 模型推理
    pre_img = model.preprocess(image)
    pred = model.predict(pre_img)
    det = pred[0]

    if det is not None and len(det):
        det_info = model.postprocess(pred)
        for info in det_info:
            # 坐标转换回原始尺寸
            if h != 640 or w != 640:
                # 转换bbox坐标
                x1, y1, x2, y2 = info['bbox']
                info['bbox'] = [
                    (x1 - pad_left) / scale,
                    (y1 - pad_top) / scale,
                    (x2 - pad_left) / scale,
                    (y2 - pad_top) / scale
                ]

                # 转换mask坐标
                if info['mask'] is not None:
                    adjusted_mask = []
                    for polygon in info['mask']:
                        adjusted = []
                        for (x, y) in polygon:
                            adj_x = (x - pad_left) / scale
                            adj_y = (y - pad_top) / scale
                            adjusted.append([adj_x, adj_y])
                        adjusted_mask.append(np.array(adjusted))
                    info['mask'] = adjusted_mask

            # 在原始图像上绘制
            original_image, _ = draw_detections(original_image, info)

    return original_image


if __name__ == "__main__":
    model = Web_Detector()
    model.load_model(r'C:\Under_water_rebar\水下钢筋\yolo11-seg\codelast20241025\code\runs\detect\train_v11LIME_light2\weights\best.pt')

    image = cv2.imread(r'C:\Under_water_rebar\final\all_light_data_split\2.jpg')
    if image is not None:
        result = process_frame(model, image)
        cv2.imshow('Result', result)
        cv2.waitKey(0)
    else:
        print("图像读取失败")