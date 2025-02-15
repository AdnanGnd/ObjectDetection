def yolo_to_bbox(yolo_bbox, image_width, image_height):
    """
    Convert YOLO format [class_id, x_center, y_center, width, height] to [x_min, y_min, x_max, y_max].
    """
    class_id, x_center, y_center, width, height = yolo_bbox
    x_min = (x_center - width / 2) * image_width
    y_min = (y_center - height / 2) * image_height
    x_max = (x_center + width / 2) * image_width
    y_max = (y_center + height / 2) * image_height
    return [x_min, y_min, x_max, y_max]