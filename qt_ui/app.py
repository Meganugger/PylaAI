import os
import sys

if os.name == "nt":
    os.environ.setdefault("QT_QPA_PLATFORM", "windows:dpiawareness=1")

from PySide6.QtCore import QUrl
from PySide6.QtGui import QIcon
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication

from qt_ui.bridge import QtBridge


def run_qt_app(version_str, brawlers, pyla_main_fn, login_fn=None, saved_brawler_data=None):
    app = QApplication.instance() or QApplication(sys.argv)
    QQuickStyle.setStyle("Basic")
    app.setApplicationName("PylaAI")

    icon_path = os.path.abspath(os.path.join("api", "assets", "brawler_icons", "8bit.png"))
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    bridge = QtBridge(
        version_str=version_str,
        brawlers=brawlers,
        pyla_main_fn=pyla_main_fn,
        login_fn=login_fn,
        saved_brawler_data=saved_brawler_data or [],
    )

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("backend", bridge)

    qml_path = os.path.abspath(os.path.join("qt_ui", "qml", "Main.qml"))
    engine.load(QUrl.fromLocalFile(qml_path))
    if not engine.rootObjects():
        raise RuntimeError(f"Failed to load QML UI from {qml_path}")

    app.aboutToQuit.connect(bridge.on_app_about_to_quit)
    return app.exec()
