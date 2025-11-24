
PRIMITIVE_LIST = [
    "plane",
    "cylinder",
    "others",
    "cone",
    "fillet",
    "chamfer",
    "sphere",
    "torus",
    "unknown"
]

PLANE_COLOR    = [255, 192, 0]
CYLINDER_COLOR = [0, 176, 240]
OTHERS_COLOR   = [225, 225, 225]
CONE_COLOR     = [0, 32, 96]
FILLET_COLOR   = [232, 119, 34]
CHAMFER_COLOR  = [112, 48, 160]
SPHERE_COLOR   = [146, 208, 80]
TORUS_COLOR    = [150, 158, 0]
UNKNOWN_COLOR  = [255, 255, 255]

# ==========================================================================================================
# ==========================================================================================================

PRIMITIVE_NAME_TO_COLOR = {
    "plane":    PLANE_COLOR,
    "cylinder": CYLINDER_COLOR,
    "others":   OTHERS_COLOR,
    "cone":     CONE_COLOR,
    "fillet":   FILLET_COLOR,
    "chamfer":  CHAMFER_COLOR,
    "sphere":   SPHERE_COLOR,
    "torus":    TORUS_COLOR,
    "unknown":  UNKNOWN_COLOR
}

PRIMITIVE_COLOR_TO_NAME = {tuple(v): k for k, v in PRIMITIVE_NAME_TO_COLOR.items()}

def prim_id_to_name(prim_id):
    return PRIMITIVE_LIST[prim_id]

def prim_name_to_id(prim_name):
    return PRIMITIVE_LIST.index(prim_name)

def prim_color_to_name(prim_color):
    prim_color = prim_color.tolist() if hasattr(prim_color, 'tolist') else prim_color
    if prim_color not in PRIMITIVE_COLOR_TO_NAME:
        print("Warning: Unknown primitive color:", prim_color)
        return "unknown"
    return PRIMITIVE_COLOR_TO_NAME[prim_color]

def prim_name_to_color(prim_name):
    if prim_name not in PRIMITIVE_NAME_TO_COLOR:
        print("Warning: Unknown primitive name:", prim_name)
        return UNKNOWN_COLOR
    return PRIMITIVE_NAME_TO_COLOR[prim_name]