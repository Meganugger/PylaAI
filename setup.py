from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).parent.resolve()
README = (ROOT / "README.md").read_text(encoding="utf-8")

SCRCPY_SOURCE = (
    "scrcpy-client @ "
    "git+https://github.com/leng-yue/py-scrcpy-client.git@"
    "f5ddaef4aa471d93f9af5f7559023f0b6a531ec9"
)

INSTALL_REQUIRES = [
    "aiohttp>=3.9.0",
    "bettercam>=1.0.0; platform_system == 'Windows'",
    "customtkinter>=5.2.0",
    "discord.py>=2.3.2",
    "easyocr>=1.7.2",
    "google-play-scraper>=1.2.7",
    "numpy>=1.24,<2",
    "opencv-python>=4.8,<5",
    "packaging>=23.1",
    "Pillow>=10,<11",
    "pyautogui>=0.9.54",
    "pywin32>=311; platform_system == 'Windows'",
    "requests>=2.31.0",
    "setuptools<81",
    "toml>=0.10.2",
    SCRCPY_SOURCE,
]

PY_MODULES = [
    "brawlstars_api",
    "detect",
    "lobby_automation",
    "main",
    "play",
    "runtime_threads",
    "stage_manager",
    "time_management",
    "trophy_observer",
    "utils",
    "window_controller",
]


setup(
    name="PylaAI",
    version="1.0.0",
    description="Windows Python automation project for Brawl Stars development workflows.",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.10,<3.11",
    packages=find_namespace_packages(
        include=["api*", "gui*", "state_finder*", "typization*"],
        exclude=["tests", "tests.*"],
    ),
    py_modules=PY_MODULES,
    include_package_data=True,
    package_data={
        "api": ["assets/brawler_icons/*.png", "assets/brawler_icons2/*.png"],
        "state_finder": ["images_to_detect/*"],
    },
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "cpu": ["onnxruntime>=1.17,<2"],
        "directml": [
            "onnxruntime-directml>=1.17,<2; platform_system == 'Windows'"
        ],
        "cuda": [
            "onnxruntime-gpu>=1.21,<2",
            "nvidia-cuda-runtime-cu12==12.4.*; platform_system == 'Windows'",
            "nvidia-cublas-cu12==12.4.*; platform_system == 'Windows'",
            "nvidia-cufft-cu12>=11,<12; platform_system == 'Windows'",
            "nvidia-cudnn-cu12==9.20.*; platform_system == 'Windows'",
        ],
        "dev": ["pytest>=8.0"],
    },
    classifiers=[
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    license="Proprietary",
    zip_safe=False,
)
