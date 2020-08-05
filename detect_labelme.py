import argparse
from sys import platform
import json
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from base64 import b64encode
from json import dumps
import os
import imageio
import glob
from PIL import Image
import os.path as osp

global count_labelme


class img_to_json(object):
    """
        这个类是用来将图像数据转化成json文件的，方便下一步的处理。主要是为了获取
        图像的字符串信息
    """

    def __init__(self, process_img_path="",
                 img_type=".png",
                 out_file_path="./",
                 out_file_type=".json"):
        """
        :param process_img_path: 待处理图片的路径
        :param img_type: 待处理图片的类型
        :param out_file_path: 输出文件的路径
        :param out_file_type: 输出文件的类型
        使用glob从指定路径中获取所有的img_type的图片
        """
        self.process_img = process_img_path
        self.out_file = out_file_path
        self.out_file_type = out_file_type
        self.img_type = img_type

    def en_decode(self):
        """
        对获取的图像数据进行编码，解码后并存储到指定文件，保存为json文件
        :return: null
        """
        # print('-' * 30)
        # print("运行 Encode--->Decode\nStart process.....\nPlease wait a moment")
        # print('-' * 30)
        """
        Start process.....   Please wait a moment
        """
        """filepath, shotname, extension, tempfilename:目标文件所在路径，文件名，文件后缀,文件名+文件后缀"""

        def capture_file_info(filename):
            (filepath, tempfilename) = os.path.split(filename)
            (shotname, extension) = os.path.splitext(tempfilename)
            return filepath, shotname, extension, tempfilename

        ENCODING = 'utf-8'  # 编码形式为utf-8

        # SCRIPT_NAME, IMAGE_NAME, JSON_NAME = argv  # 获得文件名参数

        imgname = self.process_img  # 所有图片的形成的列表信息
        img_size = Image.open(imgname).size

        # img_number = capture_file_info(img)[1]
        # imgs = sorted(img,key=lambda )

        out_file_path = self.out_file

        # imgtype = self.img_type

        out_file_type = self.out_file_type
        print("待处理的图片的数量:", len(imgname))
        if len(imgname) == 0:
            print("There was nothing under the specified path.")
            return 0
        # for imgname in img:
        # midname = imgname[imgname.rindex("\\"):imgname.rindex("." + imgtype)]
        midname = capture_file_info(imgname)[1]  # midname:图片的名称，不带后缀名
        IMAGE_NAME = imgname
        # IMAGE_NAME = midname + imgtype
        JSON_NAME = midname + out_file_type
        # 读取二进制图片，获得原始字节码，注意 'rb'
        with open(IMAGE_NAME, 'rb') as jpg_file:
            byte_content = jpg_file.read()
        # 把原始字节码编码成 base64 字节码
        base64_bytes = b64encode(byte_content)
        # 将 base64 字节码解码成 utf-8 格式的字符串
        base64_string = base64_bytes.decode(ENCODING)
        # 用字典的形式保存数据
        """raw_data:用来存放加入特性的数据，img_raw_data:用来存放不加入特性的数据，只有图片的字符串数据"""
        # raw_data = {}
        # raw_data["name"] = IMAGE_NAME
        # raw_data["image_base64_string"] = base64_string
        img_raw_data = {}
        img_raw_data = base64_string
        # 将字典变成 json 格式，indent =2:表示缩进为 2 个空格
        # json_data = dumps(raw_data)
        json_img_data = dumps(img_raw_data)
        # 将 json 格式的数据保存到指定的文件中
        # print(json_img_data)
        # with open(out_file_path+JSON_NAME, 'w') as json_file:
        #     json_file.write(json_img_data)

        return imgname, json_img_data, img_size


class Params():
    """Class that loads hyperparameters from a json file.
        Example:
        ```
        params = Params(json_path)
        print(params.learning_rate)
        params.learning_rate = 0.5  # change the value of learning_rate in params
        ```
        """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)  # 将json格式数据转换为字典
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)  # indent缩进级别进行漂亮打印

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property  # Python内置的@property装饰器就是负责把一个方法变成属性调用的
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def detect(save_txt=False, save_img=False, json_str=None, source_file=None):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    global count_labelme
    # json format

    if not os.path.exists(source_file.split(".")[0]):
        os.mkdir(source_file.split(".")[0])

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    count_labelme = 0
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        if count_labelme % 20 == 0:
                parameters = {"version": "3.16.7",
                  "flags": {},
                  "shapes": [],
                  "lineColor": [
                      0,
                      255,
                      0,
                      128
                  ],
                  "fillColor": [
                      255,
                      0,
                      0,
                      128
                  ],
                  "imagePath": None,
                  "imageData": None,
                  "imageHeight": None,
                  "imageWidth": None
                  }
                json_str = json.dumps(parameters, indent=4)

                with open('./' + source_file.split(".")[0] + '/' + source_file.split(".")[0] + '%05d.json' % count_labelme, 'w') as f:
                    f.write(json_str)  # 将json_str写到文件中
                params = Params(
                    './' + source_file.split(".")[0] + '/' + source_file.split(".")[0] + '%05d.json' % count_labelme)
                imageio.imsave(
                    './' + source_file.split(".")[0] + '/' + source_file.split(".")[0] + '%05d.jpg' % count_labelme,
                    im0s[:, :, ::-1])





        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        # Apply
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s



        save_path = str(Path(out) / Path(p).name)
        s += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, classes[int(c)])  # add to string
            if count_labelme % 20 == 0:  # if count ==15
                # Write results
                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                    # json file
                    print(xyxy[0].cpu().data.numpy())

                    newdata = {
                        "label": "person",
                        "line_color": None,
                        "fill_color": None,
                        "points": [
                            [
                                float(xyxy[0].cpu().data.numpy()),
                                float(xyxy[1].cpu().data.numpy())
                            ],
                            [
                                float(xyxy[2].cpu().data.numpy()),
                                float(xyxy[3].cpu().data.numpy())
                            ]
                        ],
                        "shape_type": "rectangle",
                        "flags": {}
                    }
                    params.shapes.append(newdata)  # 修改json中的数据
                trans = img_to_json(
                    process_img_path='./' + source_file.split(".")[0] + '/' + source_file.split(".")[
                        0] + '%05d.jpg' % count_labelme)
                imagePath, decode_data, img_size = trans.en_decode()
                #print(imagePath,type(decode_data),type(img_size))
                params.imagePath = imagePath.split("/")[-1]
                params.imageData = decode_data
                params.imageHeight = img_size[1]
                params.imageWidth = img_size[0]
        if count_labelme % 20 == 0:  # if count ==15
        	params.save('./' + source_file.split(".")[0] + '/' + source_file.split(".")[0] + '%05d.json' % count_labelme)  # 将修改后的数据保存
        f.close()
        count_labelme = count_labelme + 1
    print('%sDone. (%.3fs)' % (s, time.time() - t))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/backup150.pt', help='path to weights file')
    parser.add_argument('--source', dest='source', type=str, default='data/samples',
                        help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    source = opt.source

    with torch.no_grad():
        detect(source_file=source)

