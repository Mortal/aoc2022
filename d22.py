# "if 1": use my input - "if 0": use sample input
if 0:
    with open("d22.txt") as fp:
        the_input = fp.read()
else:
    the_input = """
        ...#
        .#..
        #...
        ....
...#.......#
........#...
..#....#....
..........#.
        ...#....
        .....#..
        .#......
        ......#.

10R5L5R10L4R5L5
"""

themap_str, directions_str = the_input.strip("\n").split("\n\n")

directions: list[str] = []
for directions_str_1 in directions_str.split("L"):
    if directions:
        directions.append("L")
    for directions_str_2 in directions_str_1.split("R"):
        if directions and directions[-1] != "L":
            directions.append("R")
        directions.append(directions_str_2)

themap = themap_str.split("\n")
area = sum(v != " " for line in themap for v in line)
facesize = int((area / 6) ** 0.5)
thenet = [row[::facesize] for row in themap[::facesize]]
first_net_location = next(
    (i, j) for i, line in enumerate(thenet) for j, v in enumerate(line) if v != " "
)

Vec2 = tuple[int, int]
Vec3 = tuple[int, int, int]
Quad3 = tuple[Vec3, Vec3, Vec3, Vec3]

RIGHT, DOWN, LEFT, UP = range(4)
fourneighbors = {RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1), UP: (-1, 0)}

# top-right, bottom-right, bottom-left, top-left
TOP = ((1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0))
BOT = ((1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1))
VERTEX_NAME = dict(zip([*TOP, *BOT], "ABCDEFGH"))


def cross(o: Vec3, u: Vec3, v: Vec3) -> Vec3:
    a = (u[0] - o[0], u[1] - o[1], u[2] - o[2])
    b = (v[0] - o[0], v[1] - o[1], v[2] - o[2])
    s0 = a[1] * b[2] - a[2] * b[1]
    s1 = a[2] * b[0] - a[0] * b[2]
    s2 = a[0] * b[1] - a[1] * b[0]
    return (o[0] + s0, o[1] + s1, o[2] + s2)


def rotate_quad(xs: Quad3, shift: int) -> Quad3:
    return (*xs[shift:], *xs[:shift])  # type: ignore


Halfedge = tuple[Vec2, int]


def draw_halfedges(
    halfedges: dict[Halfedge, Halfedge], drawing: dict[Vec2, str]
) -> dict[Vec2, str]:
    w = 8
    first = "a"

    for i, (((i1, j1), d1), ((i2, j2), d2)) in enumerate(halfedges.items()):
        for k in range(1, 7):
            drawing.setdefault((6 * i1 + k, w * j1), "|")
            drawing.setdefault((6 * i1 + k, w * j1 + w), "|")
        for k in range(1, w + 1):
            drawing.setdefault((6 * i1, w * j1 + k), "-")
            drawing.setdefault((6 * i1 + 6, w * j1 + k), "-")
        drawing[6 * i1, w * j1] = "+"
        drawing[6 * i1 + 6, w * j1] = "+"
        drawing[6 * i1 + 6, w * j1 + w] = "+"
        drawing[6 * i1, w * j1 + w] = "+"
        if d1 == RIGHT:
            drawing[6 * i1 + 2, w * j1 + w] = ">v<^"[d1]
            drawing[6 * i1 + 2, w * j1 + w - 1] = chr(ord(first) + i)
        if d2 == RIGHT:
            drawing[6 * i2 + 2, w * j2] = ">v<^"[d2]
            drawing[6 * i2 + 2, w * j2 + 1] = chr(ord(first) + i)
        if d1 == LEFT:
            drawing[6 * i1 + 4, w * j1] = ">v<^"[d1]
            drawing[6 * i1 + 4, w * j1 + 1] = chr(ord(first) + i)
        if d2 == LEFT:
            drawing[6 * i2 + 4, w * j2 + w] = ">v<^"[d2]
            drawing[6 * i2 + 4, w * j2 + w - 1] = chr(ord(first) + i)
        if d1 == UP:
            drawing[6 * i1, w * j1 + w // 2 - 1] = ">v<^"[d1]
            drawing[6 * i1 + 1, w * j1 + w // 2 - 1] = chr(ord(first) + i)
        if d2 == UP:
            drawing[6 * i2 + 6, w * j2 + w // 2 - 1] = ">v<^"[d2]
            drawing[6 * i2 + 5, w * j2 + w // 2 - 1] = chr(ord(first) + i)
        if d1 == DOWN:
            drawing[6 * i1 + 6, w * j1 + w // 2 + 1] = ">v<^"[d1]
            drawing[6 * i1 + 5, w * j1 + w // 2 + 1] = chr(ord(first) + i)
        if d2 == DOWN:
            drawing[6 * i2, w * j2 + w // 2 + 1] = ">v<^"[d2]
            drawing[6 * i2 + 1, w * j2 + w // 2 + 1] = chr(ord(first) + i)
    return drawing


def draw_faces(faces: dict[Vec2, Quad3]) -> dict[Vec2, str]:
    drawing: dict[Vec2, str] = {}
    w = 8

    def corner_name(corner: Vec3) -> str:
        i, j, k = corner
        return chr(ord("A") + i + 2 * j + 4 * k)

    for face_index, ((i, j), (a, b, c, d)) in enumerate(faces.items()):
        for k in range(1, 7):
            drawing.setdefault((6 * i + k, w * j), "|")
            drawing.setdefault((6 * i + k, w * j + w), "|")
        for k in range(1, w + 1):
            drawing.setdefault((6 * i, w * j + k), "-")
            drawing.setdefault((6 * i + 6, w * j + k), "-")
        drawing[6 * i, w * j] = "+"
        drawing[6 * i + 6, w * j] = "+"
        drawing[6 * i + 6, w * j + w] = "+"
        drawing[6 * i, w * j + w] = "+"
        drawing[6 * i + 3, w * j + w // 2] = chr(ord("1") + face_index)
        drawing[6 * i + 1, w * j + w - 1] = corner_name(a)
        drawing[6 * i + 5, w * j + w - 1] = corner_name(b)
        drawing[6 * i + 5, w * j + 1] = corner_name(c)
        drawing[6 * i + 1, w * j + 1] = corner_name(d)

    return drawing


def print_drawing(drawing: dict[Vec2, str]) -> None:
    ii, jj = 0, 0
    output: list[str] = []
    for (i, j), ch in sorted(drawing.items()):
        if i != ii:
            output.append("\n" * (i - ii))
            ii, jj = i, 0
        if j != jj:
            output.append(" " * (j - jj))
            jj = j
        output.append(ch)
        jj += 1
    print("".join(output))


def fold1(thenet: list[str]) -> dict[Halfedge, Halfedge]:
    # Construct half-edges for part 1
    halfedges: dict[Halfedge, Halfedge] = {}
    ups: list[list[int]] = []
    lefts: list[list[int]] = []
    n = 0
    for i, row in enumerate(thenet):
        ups.append([])
        lefts.append([])
        for j, v in enumerate(row):
            if v == " ":
                lefts[-1].append(j + 1)
                ups[-1].append(i + 1)
                continue
            n += 1
            if j == 0:
                lefts[-1].append(0)
            else:
                lefts[-1].append(lefts[-1][-1])
            if i == 0:
                ups[-1].append(0)
            elif j < len(ups[i - 1]):
                ups[-1].append(ups[i - 1][j])
            else:
                ups[-1].append(i)
            if j + 1 == len(row) or row[j + 1] == " ":
                halfedges[(i, j), RIGHT] = (i, lefts[-1][-1]), RIGHT
                halfedges[(i, lefts[-1][-1]), LEFT] = (i, j), LEFT
            else:
                halfedges[(i, j), RIGHT] = (i, j + 1), RIGHT
                halfedges[(i, j + 1), LEFT] = (i, j), LEFT
            if (
                i + 1 == len(thenet)
                or j >= len(thenet[i + 1])
                or thenet[i + 1][j] == " "
            ):
                halfedges[(i, j), DOWN] = (ups[-1][-1], j), DOWN
                halfedges[(ups[-1][-1], j), UP] = (i, j), UP
            else:
                halfedges[(i, j), DOWN] = (i + 1, j), DOWN
                halfedges[(i + 1, j), UP] = (i, j), UP

    assert len(halfedges) == 4 * n, (n, len(halfedges))
    print_drawing(draw_halfedges(halfedges, {}))
    return halfedges


def fold2(thenet: list[str], first_net_location: Vec2) -> dict[Halfedge, Halfedge]:
    # Construct half-edges for part 2
    n = len(thenet)
    faces = {first_net_location: TOP}
    dfs = [first_net_location]
    while dfs:
        i, j = dfs.pop()
        for e in range(4):
            di, dj = fourneighbors[e]
            ii, jj = i + di, j + dj
            if not (0 <= ii < len(thenet) and 0 <= jj < len(thenet[ii])):
                continue
            if thenet[ii][jj] == " " or (ii, jj) in faces:
                continue
            a, b, c, d = rotate_quad(faces[i, j], e)
            faces[ii, jj] = rotate_quad((cross(a, b, d), cross(b, c, a), b, a), -e)
            dfs.append((ii, jj))
    build_halfedges = {
        (v1, v2): (loc, direction)
        for loc, face in faces.items()
        for direction, (v1, v2) in enumerate(zip(face, rotate_quad(face, 1)))
    }
    assert len(build_halfedges) == 24
    halfedges = {
        (loc1, dir1): (loc2, (dir2 + 2) % 4)
        for (u, v), (loc1, dir1) in build_halfedges.items()
        for loc2, dir2 in [build_halfedges[v, u]]
    }
    assert len(halfedges) == 24
    print_drawing(draw_halfedges(halfedges, draw_faces(faces)))
    return halfedges


# face: 2d coordinate into thenet
# facing: RIGHT or DOWN or LEFT or UP
# i,j: face-local 2d coordinate into `themap`, relative to `face`

Position = tuple[Vec2, int, int, int]
start_position: Position = first_net_location, RIGHT, 0, 0


def onestep(halfedges: dict[Halfedge, Halfedge], pos: Position) -> Position:
    face, facing, i, j = pos
    # 0/1 indicators for going in certain directions
    downleft1 = (facing == DOWN) + (facing == LEFT)
    # -1/0/1 indicators for going in a certain direction
    goright1 = (facing == RIGHT) - (facing == LEFT)
    godown1 = (facing == DOWN) - (facing == UP)
    ni, nj = i + godown1, j + goright1
    if 0 <= ni < facesize and 0 <= nj < facesize:
        return face, facing, ni, nj
    # k: local position on edge, no.cells from start of edge
    # when going clockwise around the face
    k = -godown1 * j + goright1 * i + downleft1 * (facesize - 1)
    assert 0 <= k < facesize
    # Follow half-edge to a new face
    nface, nfacing = halfedges[face, facing]
    # 0/1 indicators for going in certain directions
    downleft2 = (nfacing == DOWN) + (nfacing == LEFT)
    upleft2 = (nfacing == UP) + (nfacing == LEFT)
    # -1/0/1 indicators for going in a certain direction
    goright2 = (nfacing == RIGHT) - (nfacing == LEFT)
    godown2 = (nfacing == DOWN) - (nfacing == UP)
    # Compute face-local coordinate on new face
    ni = upleft2 * (facesize - 1) + k * goright2
    nj = downleft2 * (facesize - 1) + k * -godown2
    assert 0 <= ni < facesize
    assert 0 <= nj < facesize
    return nface, nfacing, ni, nj


def solve(halfedges: dict[Halfedge, Halfedge]) -> None:
    face, facing, i, j = start_position
    lastdir = {(face[0] * facesize + i, face[1] * facesize + j): facing}
    for direction in directions:
        if direction == "L":
            facing = (facing - 1) % 4
            lastdir[face[0] * facesize + i, face[1] * facesize + j] = facing
        elif direction == "R":
            facing = (facing + 1) % 4
            lastdir[face[0] * facesize + i, face[1] * facesize + j] = facing
        else:
            nsteps = int(direction)
            for _ in range(nsteps):
                nface, nfacing, ni, nj = onestep(halfedges, (face, facing, i, j))
                nii = nface[0] * facesize + ni
                njj = nface[1] * facesize + nj
                assert 0 <= nii < len(themap)
                assert 0 <= njj < len(themap[nii])
                assert themap[nii][njj] != " "
                if themap[nii][njj] == "#":
                    # Hit wall - don't update face,facing,i,j
                    break
                face, facing, i, j = nface, nfacing, ni, nj
                lastdir[face[0] * facesize + i, face[1] * facesize + j] = facing
    print("Final position:", (face, facing, i, j))
    ii = face[0] * facesize + i
    jj = face[1] * facesize + j
    for i, row in enumerate(themap):
        assert len(row) % facesize == 0
        print(
            "".join(
                ">v<^"[lastdir[i, j]] if (i, j) in lastdir else v
                for j, v in enumerate(row)
            )
        )
    print(f"Row {ii + 1}, column {jj + 1}, facing {facing}")
    print(f"1000 * {ii + 1} + 4 * {jj + 1} + {facing} =")
    print(1000 * (ii + 1) + 4 * (jj + 1) + facing)


print("============= PART 1 =============")
solve(fold1(thenet))
print("============= PART 2 =============")
solve(fold2(thenet, first_net_location))
