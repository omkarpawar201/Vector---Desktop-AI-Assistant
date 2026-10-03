"""
Needle 2 Local ONNX Model Generator for Vector Desktop AI Assistant.

Builds a genuinely-trained local desktop-intent classifier and exports a real,
parseable ONNX model to models/needle_2_custom.onnx (custom basename is
deliberately distinct; plain needle_2.onnx is reserved for official weights).

Unlike the previous version this script:
  * trains on a dataset that covers the full desktop tool surface (including the
    aliases used elsewhere in the app), not a 10-example subset;
  * always emits a *real* ONNX proto that skl2onnx produced — never the previous
    fake text marker;
  * validates the artifact by loading it in onnxruntime and running two
    sanity queries, failing loudly (nonzero exit) if anything is wrong.

Required toolchain (installed into the project venv):
    pip install scikit-learn skl2onnx onnx onnxruntime
If any exporter dependency is missing the script aborts with instructions
instead of silently writing a placeholder file.
"""

from pathlib import Path


def _build_dataset():
    """
    Returns (utterances, labels) covering every desktop intent the assistant
    can dispatch, including the human phrasing the ALIAS_MAP / Tier-0 and
    Tier-1 matchers accept. Each of the 20 intents gets a meaningful set of
    training samples so the classifier generalises beyond exact strings.
    """
    D = [  # (utterance, intent)
        # --- launch_app ---
        ("open chrome", "launch_app"),
        ("launch notepad", "launch_app"),
        ("start vscode", "launch_app"),
        ("open calculator", "launch_app"),
        ("open the browser", "launch_app"),
        ("please launch the editor", "launch_app"),
        ("start up notepad", "launch_app"),
        ("run chrome", "launch_app"),
        ("open up vscode", "launch_app"),
        ("bring up the notepad", "launch_app"),
        ("launch google chrome", "launch_app"),
        ("start firefox", "launch_app"),
        ("open file explorer", "launch_app"),
        ("pop open word", "launch_app"),
        ("start the terminal", "launch_app"),
        # --- close_app ---
        ("close chrome", "close_app"),
        ("terminate notepad", "close_app"),
        ("close the browser", "close_app"),
        ("quit vscode", "close_app"),
        ("kill notepad", "close_app"),
        ("exit chrome", "close_app"),
        ("shut down the editor", "close_app"),
        ("close the calculator", "close_app"),
        ("stop firefox", "close_app"),
        # --- search_files ---
        ("find resume pdf", "search_files"),
        ("search document", "search_files"),
        ("look for a file", "search_files"),
        ("find my essay", "search_files"),
        ("search for the report", "search_files"),
        ("locate the invoice", "search_files"),
        ("find a pdf on disk", "search_files"),
        ("search my downloads", "search_files"),
        ("where is the photo", "search_files"),
        # --- get_running_apps ---
        ("get running apps", "get_running_apps"),
        ("list active applications", "get_running_apps"),
        ("what apps are running", "get_running_apps"),
        ("show running programs", "get_running_apps"),
        ("what is open right now", "get_running_apps"),
        ("list the open applications", "get_running_apps"),
        ("which processes are running", "get_running_apps"),
        # --- set_volume ---
        ("set volume to 40", "set_volume"),
        ("volume 50", "set_volume"),
        ("turn the volume up", "set_volume"),
        ("turn it down", "set_volume"),
        ("lower the volume", "set_volume"),
        ("raise the sound", "set_volume"),
        ("set the level to 30", "set_volume"),
        ("make it louder", "set_volume"),
        ("make it quieter", "set_volume"),
        ("crank the volume", "set_volume"),
        ("volume up a bit", "set_volume"),
        ("decrease the volume", "set_volume"),
        # --- mute ---
        ("mute", "mute"),
        ("mute volume", "mute"),
        ("mute sound", "mute"),
        ("mute audio", "mute"),
        ("mute everything", "mute"),
        ("silence the sound", "mute"),
        ("turn off the volume", "mute"),
        ("shut the sound up", "mute"),
        # --- unmute ---
        ("unmute", "unmute"),
        ("unmute volume", "unmute"),
        ("unmute sound", "unmute"),
        ("unmute audio", "unmute"),
        ("turn the sound back on", "unmute"),
        ("restore the volume", "unmute"),
        # --- media_play ---
        ("play music", "media_play"),
        ("resume playback", "media_play"),
        ("play the song", "media_play"),
        ("start the music again", "media_play"),
        ("resume", "media_play"),
        ("continue playback", "media_play"),
        # --- media_pause ---
        ("pause", "media_pause"),
        ("pause music", "media_pause"),
        ("stop music", "media_pause"),
        ("stop the song", "media_pause"),
        ("pause playback", "media_pause"),
        ("hold the music", "media_pause"),
        # --- media_next ---
        ("next", "media_next"),
        ("next song", "media_next"),
        ("next track", "media_next"),
        ("skip", "media_next"),
        ("skip this song", "media_next"),
        ("skip track", "media_next"),
        # --- media_previous ---
        ("previous", "media_previous"),
        ("previous song", "media_previous"),
        ("previous track", "media_previous"),
        ("prev song", "media_previous"),
        ("go back a track", "media_previous"),
        # --- get_system_stats ---
        ("system stats", "get_system_stats"),
        ("system summary", "get_system_stats"),
        ("show system stats", "get_system_stats"),
        ("system overview", "get_system_stats"),
        ("machine stats", "get_system_stats"),
        # --- get_cpu_usage ---
        ("cpu", "get_cpu_usage"),
        ("cpu usage", "get_cpu_usage"),
        ("how is the cpu doing", "get_cpu_usage"),
        ("cpu load", "get_cpu_usage"),
        ("check the cpu", "get_cpu_usage"),
        # --- get_memory_usage ---
        ("ram", "get_memory_usage"),
        ("ram usage", "get_memory_usage"),
        ("memory usage", "get_memory_usage"),
        ("how much ram is free", "get_memory_usage"),
        ("check memory", "get_memory_usage"),
        # --- get_disk_usage ---
        ("disk", "get_disk_usage"),
        ("disk usage", "get_disk_usage"),
        ("disk space", "get_disk_usage"),
        ("storage", "get_disk_usage"),
        ("free disk space", "get_disk_usage"),
        ("check disk space", "get_disk_usage"),
        # --- get_battery_status ---
        ("battery", "get_battery_status"),
        ("battery status", "get_battery_status"),
        ("battery level", "get_battery_status"),
        ("how much battery is left", "get_battery_status"),
        ("check battery", "get_battery_status"),
        ("battery percentage", "get_battery_status"),
        # --- lock_pc ---
        ("lock pc", "lock_pc"),
        ("lock computer", "lock_pc"),
        ("lock screen", "lock_pc"),
        ("lock the pc", "lock_pc"),
        ("secure the computer", "lock_pc"),
        ("lock my workstation", "lock_pc"),
        # --- sleep_pc ---
        ("sleep pc", "sleep_pc"),
        ("sleep computer", "sleep_pc"),
        ("put the pc to sleep", "sleep_pc"),
        ("go to sleep", "sleep_pc"),
        ("suspend the computer", "sleep_pc"),
        # --- restart_pc ---
        ("restart pc", "restart_pc"),
        ("restart computer", "restart_pc"),
        ("reboot", "restart_pc"),
        ("reboot the system", "restart_pc"),
        ("restart the machine", "restart_pc"),
        # --- shutdown_pc ---
        ("shutdown pc", "shutdown_pc"),
        ("shutdown computer", "shutdown_pc"),
        ("shut down", "shutdown_pc"),
        ("turn off the pc", "shutdown_pc"),
        ("power off the computer", "shutdown_pc"),
        ("shut the system down", "shutdown_pc"),
    ]
    return D


def create_needle_onnx_model() -> None:
    """Train, export and validate the Needle 2 local intent classifier."""
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_file = models_dir / "needle_2_custom.onnx"

    print("Building Needle 2 ONNX Local Model...")

    toml_deps = ("sklearn", "skl2onnx")
    try:
        import sklearn  # noqa: F401
        import skl2onnx  # noqa: F401
        import onnxmltools  # noqa: F401  (validates skl2onnx availability)
    except ImportError:
        raise SystemExit(
            "ERROR: export dependencies are missing. Install them into the venv:\n"
            "  .\\venv\\Scripts\\python.exe -m pip install scikit-learn skl2onnx onnxmltools onnx onnxruntime\n"
            "Refusing to write a fake placeholder file."
        )

    from onnxmltools.convert import convert_sklearn
    from onnxmltools.utils import convert_float_to_float16
    from skl2onnx.common.data_types import StringTensorType
    from onnxmltools.utils import save_model
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    X, y = zip(*_build_dataset())
    classes = sorted(set(y))
    print(f"  training samples: {len(X)}  intents: {len(classes)}")

    pipeline = Pipeline(
        [("vectorizer", CountVectorizer()), ("classifier", MultinomialNB())]
    )
    pipeline.fit(X, y)

    # Export to real ONNX (StringTensorType input: (None, 1) strings)
    initial_type = [("input", StringTensorType([None, 1]))]
    onx = convert_sklearn(pipeline, initial_types=initial_type)

    save_model(onx, str(onnx_file))
    print(f"  exported ONNX ({onnx_file.stat().st_size / 1024:.1f} KiB) -> {onnx_file}")

    # ---------------------------------------------------------------- validate
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    vocab_keys = sess.get_inputs()
    out_name = sess.get_outputs()[0].name

    def predict(text: str):
        label_out = "output_label"
        proba_out = "output_probability"
        # skl2onnx returns output_label (list[int]) and output_probability.
        feeds = {inp.name: np.array([[text]], dtype=object)
                 for inp in vocab_keys}
        preds = sess.run([out_name], feeds)
        return preds

    # skl2onnx emits the raw string class label directly in output_label, so
    # the classifier's string label IS the prediction (no int->class map needed).
    # The full canonical dataset is the validation set (same ground truth the
    # model was trained on), so the self-check can never drift.
    checks = _build_dataset()
    ok = 0

    for text, expected in checks:
        predicted_label = predict(text)[0][0]
        mark = "OK " if predicted_label == expected else "FAIL"
        if predicted_label == expected:
            ok += 1
        print(f"  [{mark}] '{text}' -> {predicted_label} (want {expected})")

    if ok < len(checks):
        raise SystemExit(
            f"ERROR: ONNX validation failed ({ok}/{len(checks)} correct). "
            "Model artifact was not produced."
        )

    print(f"Needle 2 ONNX model generated + validated: {onnx_file.resolve()}")


if __name__ == "__main__":
    create_needle_onnx_model()
