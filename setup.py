from cx_Freeze import setup, Executable

setup(name = "Object Detection Software",
      version="0.1",
      description="Software detects objects in real time using yolov4 dnn",
      executable=[Executable("main.py")]
    )
