from setuptools import setup


setup(
    name="svdc-preception-vlm",
    version="0.1.0",
    description="VLM-based driving-scene analysis with local webcam and ROS 2 camera entrypoints.",
    py_modules=[
        "vlm_driving_common",
        "test_with_webcam",
        "test_with_ros_camera",
    ],
    install_requires=[
        "numpy>=1.24",
        "opencv-python>=4.8",
        "Pillow>=10.0",
        "requests>=2.31",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "svdc-webcam=test_with_webcam:main",
            "svdc-ros-camera=test_with_ros_camera:main",
        ]
    },
)
