from setuptools import setup, find_packages

setup(name='yolo3_deepsort',

      version='0.1',

      url='https://github.com/GlassyWing/yolo3_deepsort',

      license='GPL 3.0',

      author='Manlier',

      author_email='dengjiaxim@gmail.com',

      description='MOT base on yolo3+deepsort',

      keywords='detections,video,image,yolo3,deepsort',

      packages=find_packages(exclude=['test', 'weights', 'config', 'data']),

      package_data={'yolo3': ['*.*']},

      long_description=open('README.md', encoding='utf-8').read(),

      zip_safe=False,

      install_requires=['torch', 'torchvision', 'opencv-python', 'matplotlib', 'terminaltables', 'tqdm'],

      )
