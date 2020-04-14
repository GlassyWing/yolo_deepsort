from setuptools import setup, find_packages

setup(name='yolo3_torch',

      version='0.1',

      url='https://github.com/GlassyWing/yolo3_torch',

      license='GPL 3.0',

      author='Manlier',

      author_email='dengjiaxim@gmail.com',

      description='yolo3 base on torch',

      keywords='detections,video,image,yolo3',

      packages=find_packages(exclude=['test', 'weights', 'config', 'data']),

      package_data={'yolo3': ['*.*']},

      long_description=open('README.md', encoding='utf-8').read(),

      zip_safe=False,

      install_requires=['torch', 'torchvision', 'matplotlib', 'opencv-python', 'terminaltables', 'tqdm'],

      )
