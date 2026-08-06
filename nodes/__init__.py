from .field_noise import FieldNoise
from .field_remap import FieldRemap
from .field_composite import FieldComposite

NODE_CLASS_MAPPINGS = {
    "FieldNoise": FieldNoise,
    "FieldRemap": FieldRemap,
    "FieldComposite": FieldComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FieldNoise": "Field Noise",
    "FieldRemap": "Field Remap",
    "FieldComposite": "Field Composite",
}
