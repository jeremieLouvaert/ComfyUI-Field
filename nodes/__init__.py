from .field_noise import FieldNoise
from .field_remap import FieldRemap
from .field_composite import FieldComposite
from .field_threshold import FieldThreshold
from .field_morphology import FieldMorphology
from .field_combine import FieldCombine
from .field_distance import FieldDistance
from .field_from_image import FieldFromImage
from .field_from_edges import FieldFromEdges
from .field_from_detail import FieldFromDetail
from .field_gradient import FieldGradient
from .field_shape import FieldShape
from .field_tile import FieldTile
from .field_warp import FieldWarp
from .field_scatter import FieldScatter

NODE_CLASS_MAPPINGS = {
    # Phase 0
    "FieldNoise": FieldNoise,
    "FieldRemap": FieldRemap,
    "FieldComposite": FieldComposite,
    # Phase 1 (a), shaping
    "FieldThreshold": FieldThreshold,
    "FieldMorphology": FieldMorphology,
    "FieldCombine": FieldCombine,
    # Phase 1 (b), distance
    "FieldDistance": FieldDistance,
    # Phase 1 (c), derive
    "FieldFromImage": FieldFromImage,
    "FieldFromEdges": FieldFromEdges,
    "FieldFromDetail": FieldFromDetail,
    # Phase 2a, analytic generators
    "FieldGradient": FieldGradient,
    "FieldShape": FieldShape,
    "FieldTile": FieldTile,
    # Phase 2b, warp + scatter
    "FieldWarp": FieldWarp,
    "FieldScatter": FieldScatter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FieldNoise": "Field Noise",
    "FieldRemap": "Field Remap",
    "FieldComposite": "Field Composite",
    "FieldThreshold": "Field Threshold",
    "FieldMorphology": "Field Morphology",
    "FieldCombine": "Field Combine",
    "FieldDistance": "Field Distance",
    "FieldFromImage": "Field From Image",
    "FieldFromEdges": "Field From Edges",
    "FieldFromDetail": "Field From Detail",
    "FieldGradient": "Field Gradient",
    "FieldShape": "Field Shape",
    "FieldTile": "Field Tile",
    "FieldWarp": "Field Warp",
    "FieldScatter": "Field Scatter",
}
