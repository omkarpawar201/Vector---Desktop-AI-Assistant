"""
Shared, non-configurable identity constants for the Vector configuration
package.

The product name and version live here rather than on
:class:`app.config.settings.Settings` so they can never be overridden through
the environment or a ``.env`` file, and rather than in
``app.config.constants`` to keep them in a plain, line-ending-neutral module
imported by the status bar and the Diagnostics report alike.
"""

APP_NAME: str = "Vector"
APP_VERSION: str = "1.0"
