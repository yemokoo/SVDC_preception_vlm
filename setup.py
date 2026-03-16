from setuptools import setup


package_name = "svdc_preception_vlm"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml", "README.md"]),
    ],
    install_requires=[
        "setuptools",
        "numpy>=1.24",
        "opencv-python>=4.8",
        "Pillow>=10.0",
        "requests>=2.31",
    ],
    zip_safe=True,
    maintainer="yemoy",
    maintainer_email="yemoy@example.com",
    description="VLM-based driving-scene analysis with local webcam and ROS 2 camera entrypoints.",
    license="TODO: License declaration",
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "svdc-webcam = svdc_preception_vlm.test_with_webcam:main",
            "svdc-ros-camera = svdc_preception_vlm.test_with_ros_camera:main",
            "webcam_node = svdc_preception_vlm.test_with_webcam:main",
            "ros_camera_node = svdc_preception_vlm.test_with_ros_camera:main",
        ],
    },
)
