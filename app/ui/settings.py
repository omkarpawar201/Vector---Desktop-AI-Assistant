"""
Settings UI configuration dialog for Vector Desktop AI Assistant.
"""

from typing import Optional
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from app.config.settings import Settings, get_settings


class SettingsDialog(QDialog):
    """
    Configuration modal for toggling API keys, feature toggles, and privacy options.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Settings — Vector Desktop AI Assistant")
        self.setFixedWidth(440)
        self.setModal(True)
        self.settings = get_settings()

        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel {
                color: #cdd6f4;
                font-size: 13px;
            }
            QLineEdit {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                color: #cdd6f4;
                padding: 6px;
            }
            QCheckBox {
                color: #cdd6f4;
                spacing: 8px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #11111b;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        form = QFormLayout()

        self.txt_gemini_key = QLineEdit(self)
        self.txt_gemini_key.setEchoMode(QLineEdit.Password)
        self.txt_gemini_key.setText(self.settings.gemini_api_key)
        form.addRow(QLabel("Gemini API Key:"), self.txt_gemini_key)

        self.chk_gemini = QCheckBox("Enable Gemini Cloud Fallback", self)
        self.chk_gemini.setChecked(self.settings.enable_gemini)
        form.addRow(self.chk_gemini)

        self.chk_voice = QCheckBox("Enable Speech Voice Pipeline", self)
        self.chk_voice.setChecked(self.settings.enable_voice)
        form.addRow(self.chk_voice)

        self.chk_confirm = QCheckBox("Require Confirmation for Dangerous Actions", self)
        self.chk_confirm.setChecked(self.settings.require_confirmation_for_dangerous)
        form.addRow(self.chk_confirm)

        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel", self)
        btn_cancel.clicked.connect(self.reject)

        btn_save = QPushButton("Save", self)
        btn_save.clicked.connect(self._on_save)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

    def _on_save(self) -> None:
        self.settings.gemini_api_key = self.txt_gemini_key.text().strip()
        self.settings.enable_gemini = self.chk_gemini.isChecked()
        self.settings.enable_voice = self.chk_voice.isChecked()
        self.settings.require_confirmation_for_dangerous = self.chk_confirm.isChecked()
        self.accept()
