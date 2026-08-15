from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Phase 2c spec 2.5: the Field Gradient ramp canvas widget, house JS
# (EphemeralPeek/Darkroom precedent) -- frontend-only, no server routes.
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[Field] Loaded, " + str(len(NODE_CLASS_MAPPINGS)) + " nodes: "
      + ", ".join(sorted(NODE_DISPLAY_NAME_MAPPINGS.values())))
