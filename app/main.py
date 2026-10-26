"""
Vector Desktop AI Assistant — Application Entry Point.
Initializes QApplication, dark mode theme, settings, database, and launches MainWindow.
"""

import argparse
import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from app.config.settings import get_settings
from app.memory.database import get_db_manager
from app.ui.diagnostics import RecentErrorLog
from app.ui.main_window import MainWindow


def _parse_args(argv: list) -> argparse.Namespace:
    """
    CLI entry-point options for the desktop assistant.
    """
    parser = argparse.ArgumentParser(
        prog="needle",
        description="Vector Desktop AI Assistant — multi-tier (regex / Needle 2 / Gemini) intent routing.",
    )
    parser.add_argument(
        "--show-tier",
        action="store_true",
        help="Print which tier (TIER_0_DIRECT / TIER_1_NEEDLE / TIER_2_GEMINI) resolved each command to the terminal.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args(sys.argv[1:])
    settings = get_settings()

    # Configure application logging early so every worker component surfaces
    # diagnostics at the configured verbosity (see DEFAULT_LOG_LEVEL / .env LOG_LEVEL).
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Attach the bounded, in-memory error sink so the Diagnostics report can
    # surface a few recent sanitized errors. Purely passive: no file, no network.
    RecentErrorLog.install()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    try:
        # Eagerly initialize the database manager so a configuration/permission
        # failure is caught at startup rather than deep in UI initialization.
        get_db_manager()
    except Exception as exc:
        logging.exception("Failed to initialize database: %s", exc)
        QMessageBox.critical(
            None,
            "Vector - Database initialization failed",
            "Vector could not initialize its database.\n\n"
            "Please check the database path and permissions in your configuration, "
            "then restart the application.",
        )
        return 1

    try:
        window = MainWindow(show_tier_in_terminal=args.show_tier)
    except Exception as exc:
        logging.exception("Failed to initialize main window: %s", exc)
        QMessageBox.critical(
            None,
            "Vector - Startup failed",
            "Vector could not start.\n\n"
            "Please check your configuration and permissions, then restart the application.",
        )
        return 1

    window.show()

    return app.exec()


if __name__ == "__main__":
    main()
