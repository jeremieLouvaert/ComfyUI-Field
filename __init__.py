from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("[Field] Loaded, " + str(len(NODE_CLASS_MAPPINGS)) + " nodes: "
      + ", ".join(sorted(NODE_DISPLAY_NAME_MAPPINGS.values())))
