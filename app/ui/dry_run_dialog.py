"""
Dry Run Plan Preview dialog for multi-step Gemini execution plans.
"""

from typing import Any, List, Optional
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DryRunDialog(QDialog):
    """
    Plan preview dialog displaying multi-step AI tool execution plans prior to running.
    """

    def __init__(self, plan_steps: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Plan Execution Preview — Vector")
        self.setFixedWidth(460)
        self.setModal(True)

        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel#heading {
                color: #89b4fa;
                font-size: 15px;
                font-weight: bold;
            }
            QListWidget {
                background-color: #181825;
                border: 1px solid #313244;
                color: #cdd6f4;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton#btn_run {
                background-color: #a6e3a1;
                color: #11111b;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton#btn_cancel {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        lbl_heading = QLabel("📋 Vector will execute the following plan:", self)
        lbl_heading.setObjectName("heading")
        layout.addWidget(lbl_heading)

        self.list_widget = QListWidget(self)
        for i, step in enumerate(plan_steps, start=1):
            self.list_widget.addItem(f"{i}. {step}")

        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel", self)
        btn_cancel.setObjectName("btn_cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_run = QPushButton("Execute Plan", self)
        btn_run.setObjectName("btn_run")
        btn_run.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_run)

        layout.addLayout(btn_layout)
