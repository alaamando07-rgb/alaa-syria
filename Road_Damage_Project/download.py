from roboflow import Roboflow

rf = Roboflow(api_key="abJJ8MISg2sPpPNOb3cK")
project = rf.workspace("alaa-cyep2").project("pothole-detection-ml2lx-izbr3")
version = project.version(1)
dataset = version.download("yolov8")