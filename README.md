# Detection-bbox-labelme-assist-annotation-by-using-yolo3-detector-automatic annotation tool
In this project, I finished the simplified project of labelme annotation function in common annotation work based on yolo3 detector. You can use this project to simplify your heavy annotation procedure by using this project.


## Tips
You can run the **detect_labelme.py** to try process the rough annotation work, then use the labelme software to adjust the annotation boudingbox postion in correct way.
before you run the **detect_labelme.py** code, the params about "--model --cfg " should fill correct in this python file!!!

whole command ：```python detect_labelme.py --source xxx.mp4```     ;defualt the frame interval is 20 frame 

If you are lucky, the assist boundingbox annotation results (json file) will appreant in your device.

Below imgs are some results:

![img annoation_result](https://raw.githubusercontent.com/Ronales/Detection-bbox-labelme-assist-annotation-by-using-yolo3-detector-/master/annotation_example.png)


## Notice 

- The detector code from the project https://github.com/ultralytics/yolov3
- The lableme version should restrict  3.16.1 
- The labelme annotation json file format :

```
{"version": "3.16.7",
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
```
