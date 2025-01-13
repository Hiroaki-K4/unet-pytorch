import cv2


def augmentation(ori_img_path, output_size, aug_data_num):
    img = cv2.imread(ori_img_path)
    h, w, c = img.shape
    print(h, w, c)
    if max(h, w) > output_size:
        # TODO: Keep aspect ratio
        img = cv2.resize(
            img, (output_size, output_size), interpolation=cv2.INTER_NEAREST
        )


if __name__ == "__main__":
    ori_img_path = "../resources/original"
    anno_img_path = "../resources/annotation"
    output_size = 572
    aug_data_num = 10
    augmentation(ori_img_path, output_size, aug_data_num)
