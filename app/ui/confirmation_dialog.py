"""
Confirmation modal dialog for CONFIRM and DANGEROUS tool operations.
"""

from typing import Any, Dict, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ConfirmationDialog(QDialog):
    """
    Explicit security modal prompting the user before executing confirmation-required or dangerous actions.
    """

    def __init__(
        self,
        tool_name: str,
        message: str,
        arguments: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Security Confirmation Required — Vector")
        self.setFixedWidth(420)
        self.setModal(True)

        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel#heading {
                color: #f38ba8;
                font-size: 15px;
                font-weight: bold;
            }
            QLabel#body {
                color: #cdd6f4;
                font-size: 13px;
            }
            QPushButton#btn_confirm {
                background-color: #f38ba8;
                color: #11111b;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton#btn_confirm:hover {
                background-color: #eba0ac;
            }
            QPushButton#btn_cancel {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton#btn_cancel:hover {
                background-color: #585b70;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        lbl_heading = QLabel(f"⚠️ Confirm Action: {tool_name}", self)
        lbl_heading.setObjectName("heading")

        desc_text = message or f"Vector requests permission to execute '{tool_name}'."
        if arguments:
            desc_text += f"\n\nArguments: {arguments}"

        lbl_body = QLabel(desc_text, self)
        lbl_body.setObjectName("body")
        lbl_body.setWordWrap(True)

        layout.addWidget(lbl_heading)
        layout.addWidget(lbl_body)

        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setObjectName("btn_cancel")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_confirm = QPushButton("Confirm & Execute", self)
        self.btn_confirm.setObjectName("btn_confirm")
        self.btn_confirm.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_confirm)

        layout.addLayout(btn_layout)
