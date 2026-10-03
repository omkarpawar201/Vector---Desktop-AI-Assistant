import json

from app.tools.registry import get_tool_registry


registry = get_tool_registry()

schemas = registry.export_schemas()

with open("data/vector_tool_schemas.json", "w", encoding="utf-8") as f:
    json.dump(schemas, f, indent=2, ensure_ascii=False)

print(f"Exported {len(schemas)} tool schemas")
print(json.dumps(schemas, indent=2, ensure_ascii=False))