from needle import Needle

MODEL = "models/needle3_vector.cact"

TOOLS = [
    {
        "name": "set_volume",
        "description": "Set system volume to a percentage.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                }
            },
            "required": ["level"],
        },
    },
    {
        "name": "get_volume",
        "description": "Get the current system volume.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mute",
        "description": "Mute system audio.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "unmute",
        "description": "Unmute system audio.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_cpu_usage",
        "description": "Get current CPU usage.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_memory_usage",
        "description": "Get current memory usage.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_disk_usage",
        "description": "Get current disk usage.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_battery_status",
        "description": "Get current battery status.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_running_apps",
        "description": "Get currently running applications.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "launch_app",
        "description": "Launch an application by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "close_app",
        "description": "Close an application by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "search_files",
        "description": "Search for files matching a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                },
                "directory": {
                    "type": "string",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "media_play",
        "description": "Start media playback.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "media_pause",
        "description": "Pause media playback.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "media_next",
        "description": "Skip to the next media track.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "media_previous",
        "description": "Go to the previous media track.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]


needle = Needle(
    weights=MODEL,
    tools=TOOLS,
)

TESTS = [
    "Set the volume to 50 percent",
    "Make volume 25",
    "What is my current volume?",
    "Mute the sound",
    "Unmute audio",

    "How much CPU am I using?",
    "How much RAM is being used?",
    "How much disk space do I have?",
    "What's my battery status?",
    "Show me the applications currently running",

    "Open Chrome",
    "Launch VS Code",
    "Close Spotify",

    "Play music",
    "Pause playback",
    "Skip this song",
    "Go to the previous track",

    "Find my PDF files",
]

for query in TESTS:
    needle.reset()

    print("\n" + "=" * 70)
    print("QUERY:", query)

    result = needle.complete(query)

    print("RESULT:")
    print(result)
