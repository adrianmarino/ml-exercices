import cv2


def proportional_width(height, scale_ratio): return height * scale_ratio


def padding(external, internal): return int(round((external - internal) / 2))


def remove_sky_and_card_board(img, board_height, scale_ratio):
    route_area_top = int(img.shape[0] / 2)
    route_area_bottom = img.shape[0] - board_height

    route_area_height = (route_area_bottom - route_area_top)
    route_area_width = proportional_width(route_area_height, scale_ratio)

    route_area_left = padding(img.shape[1], route_area_width)

    return img[route_area_top:route_area_bottom, route_area_left:-route_area_left]


def resize_and_color(img, width, height, color_mode='RGB'):
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    if color_mode == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img


def route_area(img, width, height, board_height=0, color_mode='RGB'):
    img = remove_sky_and_card_board(img, board_height, scale_ratio=width/height)
    return resize_and_color(img, width, height, color_mode)