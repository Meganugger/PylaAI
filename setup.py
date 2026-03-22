from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).parent.resolve()
README = (ROOT / "README.md").read_text(encoding="utf-8")

COMMON_DEPENDENCIES = [
    "adbutils==2.12.0",
    "aiohttp>=3.9.0",
    "av==12.3.0",
    "bettercam>=1.0.0",
    "customtkinter>=5.2.0",
    "discord.py>=2.3.2",
    "easyocr>=1.7.2",
    "google-play-scraper>=1.2.7",
    "numpy<2",
    "opencv-python>=4.8.0,<5.0.0",
    "packaging>=23.1",
    "Pillow>=10.0.0",
    "pyautogui>=0.9.54",
    "pywin32>=311; platform_system == 'Windows'",
    "requests>=2.31.0",
    "scrcpy-client==0.4.7",
    "shapely>=2.0",
    "toml>=0.10.2",
]

PY_MODULES = [
    "detect",
    "lobby_automation",
    "main",
    "play",
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
    python_requires=">=3.10,<3.13",
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
    install_requires=COMMON_DEPENDENCIES,
    extras_require={
        "cpu": ["onnxruntime>=1.17,<2"],
        "directml": [
            "onnxruntime-directml>=1.17,<2; platform_system == 'Windows'"
        ],
        "cuda": ["onnxruntime-gpu>=1.17,<2"],
        "dev": ["pytest>=8.0"],
    },
    license="Proprietary",
    zip_safe=False,
)
