import cv2
import glob

def resize_img(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(size,size), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(image_path,img)
    
all_images = glob.glob('tiny-224/*/*/*/*')
for image in all_images:
    resize_img(image, 224)