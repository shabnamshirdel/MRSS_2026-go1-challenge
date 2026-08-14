WALL_TAG_COUNT = 15
TAG_PITCH = 0.325
WALL_COORD = 2.4
TAG_HEIGHT = 0.5


def _centered_offsets() -> list[float]:
    return [(index - (WALL_TAG_COUNT - 1) / 2) * TAG_PITCH for index in range(WALL_TAG_COUNT)]


TAGS_LOC = {}
TAG_WALL_NORMALS = {}
tag_id = 0
for wall_name, normal in (
    ("front", [0.0, 1.0, 0.0]),
    ("left", [-1.0, 0.0, 0.0]),
    ("right", [1.0, 0.0, 0.0]),
    ("back", [0.0, -1.0, 0.0]),
):
    for offset in _centered_offsets():
        if wall_name == "front":
            position = [offset, WALL_COORD, TAG_HEIGHT]
        elif wall_name == "back":
            position = [offset, -WALL_COORD, TAG_HEIGHT]
        elif wall_name == "left":
            position = [-WALL_COORD, offset, TAG_HEIGHT]
        else:
            position = [WALL_COORD, offset, TAG_HEIGHT]
        TAGS_LOC[tag_id] = position
        TAG_WALL_NORMALS[tag_id] = normal.copy()
        tag_id += 1
