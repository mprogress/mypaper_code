def calculate_IoU(predicted_bound, ground_truth_bound):
    """
    computing the IoU of two boxes.
    Args:
        box: (xmin, ymin, xmax, ymax),通过左下和右上两个顶点坐标来确定矩形位置
    Return:
        IoU: IoU of box1 and box2.
    """
    pxmin, pymin, pxmax, pymax = predicted_bound
    print("预测框P的坐标是：({}, {}, {}, {})".format(pxmin, pymin, pxmax, pymax))
    gxmin, gymin, gxmax, gymax = ground_truth_bound
    print("原标记框G的坐标是：({}, {}, {}, {})".format(gxmin, gymin, gxmax, gymax))

    parea = (pxmax - pxmin) * (pymax - pymin)  # 计算P的面积
    garea = (gxmax - gxmin) * (gymax - gymin)  # 计算G的面积
    print("预测框P的面积是：{}；原标记框G的面积是：{}".format(parea, garea))

    # 求相交矩形的左下和右上顶点坐标(xmin, ymin, xmax, ymax)
    xmin = max(pxmin, gxmin)  # 得到左下顶点的横坐标
    ymin = max(pymin, gymin)  # 得到左下顶点的纵坐标
    xmax = min(pxmax, gxmax)  # 得到右上顶点的横坐标
    ymax = min(pymax, gymax)  # 得到右上顶点的纵坐标

    # 计算相交矩形的面积
    w = xmax - xmin
    h = ymax - ymin
    if w <=0 or h <= 0:
        return 0

    area = w * h  # G∩P的面积
    # area = max(0, xmax - xmin) * max(0, ymax - ymin)  # 可以用一行代码算出来相交矩形的面积
    print("G∩P的面积是：{}".format(area))

    # 并集的面积 = 两个矩形面积 - 交集面积
    IoU = area / (parea + garea - area)

    return IoU

if __name__ == '__main__':
    IoU = calculate_IoU( (1, -1, 3, 1), (0, 0, 2, 2))
    print("IoU是：{}".format(IoU))

