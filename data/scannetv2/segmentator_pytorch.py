import torch
import math

################################################################################
# Minimal Union-Find (Disjoint Set)
################################################################################
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0]*n
        self.size   = [1]*n
        self.num_sets = n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int):
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            # Union by rank
            if self.rank[rx] > self.rank[ry]:
                self.parent[ry] = rx
                self.size[rx]   += self.size[ry]
            elif self.rank[rx] < self.rank[ry]:
                self.parent[rx] = ry
                self.size[ry]   += self.size[rx]
            else:
                self.parent[ry] = rx
                self.size[rx]   += self.size[ry]
                self.rank[rx]   += 1
            self.num_sets -= 1
    
    def get_size(self, x: int) -> int:
        return self.size[self.find(x)]


class _Edge:
    __slots__ = ('a','b','w')
    def __init__(self, a: int, b: int, w: float):
        self.a = a
        self.b = b
        self.w = w


################################################################################
# Helper: Felzenszwalb-like segmentation
################################################################################
def _segment_graph(num_vertices: int, edges: list, c: float):
    """
    Sort edges by ascending weight, then for each edge in non-decreasing weight:
      - If edge weight <= threshold of both components, union them
      - threshold[new_component] = w + c / size(new_component)
    """
    edges.sort(key=lambda e: e.w)
    
    uf = _UnionFind(num_vertices)
    threshold = [c]*num_vertices  # initial threshold for each vertex
    
    for e in edges:
        a = uf.find(e.a)
        b = uf.find(e.b)
        if a != b:
            if (e.w <= threshold[a]) and (e.w <= threshold[b]):
                uf.union(a, b)
                rep = uf.find(a)
                size_rep = uf.get_size(rep)
                threshold[rep] = e.w + (c / size_rep)
    return uf


################################################################################
# Minimal cross product & lerp in Python
################################################################################
def _cross(u, v):
    cx = u[1]*v[2] - u[2]*v[1]
    cy = u[2]*v[0] - u[0]*v[2]
    cz = u[0]*v[1] - u[1]*v[0]
    norm = math.sqrt(cx*cx + cy*cy + cz*cz)
    if norm < 1e-12:
        return (0.0, 0.0, 0.0)
    return (cx/norm, cy/norm, cz/norm)

def _lerp(a, b, t: float):
    return (
        (1 - t)*a[0] + t*b[0],
        (1 - t)*a[1] + t*b[1],
        (1 - t)*a[2] + t*b[2],
    )


################################################################################
# The main Python segment_mesh function
################################################################################
def segment_mesh(vertices: torch.Tensor,
                 faces: torch.Tensor,
                 kThresh: float = 0.01,
                 segMinVerts: int = 20) -> torch.Tensor:
    """
    segment a mesh (CPU) using a Python implementation of Felzenszwalb-like graph segmentation

    Args:
        vertices (torch.Tensor): (nv, 3) float32 tensor of vertex positions
        faces    (torch.Tensor): (nf, 3) long/int tensor of triangle indices
        kThresh  (float): segmentation cluster threshold parameter 
                          (larger => larger segments)
        segMinVerts (int): minimum number of vertices per-segment,
                           enforces merging of small clusters
    
    Returns:
        index (torch.Tensor): (nv,) long tensor of cluster indices, reindexed to start at 0
    """
    assert vertices.dim() == 2 and vertices.size(1) == 3, "vertices must be (N,3)"
    assert faces.dim() == 2 and faces.size(1) == 3, "faces must be (M,3)"

    vertexCount = vertices.shape[0]
    faceCount   = faces.shape[0]

    # Move to CPU if needed
    verts_cpu = vertices.cpu()
    faces_cpu = faces.cpu()

    # Prepare containers
    points  = [None]*vertexCount  # each entry: (x,y,z)
    normals = [(0.0,0.0,0.0)]*vertexCount
    counts  = [0]*vertexCount
    edges   = []  # list of _Edge

    # Accumulate face normals into vertex normals & build edges
    for fi in range(faceCount):
        i1 = int(faces_cpu[fi,0])
        i2 = int(faces_cpu[fi,1])
        i3 = int(faces_cpu[fi,2])

        p1 = (verts_cpu[i1,0].item(), verts_cpu[i1,1].item(), verts_cpu[i1,2].item())
        p2 = (verts_cpu[i2,0].item(), verts_cpu[i2,1].item(), verts_cpu[i2,2].item())
        p3 = (verts_cpu[i3,0].item(), verts_cpu[i3,1].item(), verts_cpu[i3,2].item())

        # Assign points (like original code, though somewhat redundant)
        points[i1] = p1
        points[i2] = p2
        points[i3] = p3

        # Face normal
        v21 = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
        v31 = (p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2])
        fnormal = _cross(v21, v31)

        # Smoothly blend face normal into vertex normals
        n1 = normals[i1]
        t1 = 1.0/(counts[i1]+1.0)
        normals[i1] = _lerp(n1, fnormal, t1)

        n2 = normals[i2]
        t2 = 1.0/(counts[i2]+1.0)
        normals[i2] = _lerp(n2, fnormal, t2)

        n3 = normals[i3]
        t3 = 1.0/(counts[i3]+1.0)
        normals[i3] = _lerp(n3, fnormal, t3)

        counts[i1]+=1
        counts[i2]+=1
        counts[i3]+=1

        # Each face => 3 edges (undirected)
        edges.append(_Edge(i1, i2, 0.0))
        edges.append(_Edge(i1, i3, 0.0))
        edges.append(_Edge(i2, i3, 0.0))

    # Compute edge weights
    for e in edges:
        a = e.a
        b = e.b
        pA = points[a]
        pB = points[b]
        nA = normals[a]
        nB = normals[b]

        dx = pB[0] - pA[0]
        dy = pB[1] - pA[1]
        dz = pB[2] - pA[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 1e-12:
            e.w = 0.0
            continue

        ux, uy, uz = dx/dist, dy/dist, dz/dist
        dot  = nA[0]*nB[0] + nA[1]*nB[1] + nA[2]*nB[2]
        dot2 = nB[0]*ux   + nB[1]*uy   + nB[2]*uz
        ww   = 1.0 - dot
        if dot2 > 0.0:
            ww = ww*ww
        e.w = ww

    # Felzenszwalb-style segmentation
    uf = _segment_graph(vertexCount, edges, kThresh)

    # Merge small segments
    for e in edges:
        rA = uf.find(e.a)
        rB = uf.find(e.b)
        if rA != rB:
            if (uf.get_size(rA) < segMinVerts) or (uf.get_size(rB) < segMinVerts):
                uf.union(rA, rB)

    # Build output labels
    labels = torch.empty(vertexCount, dtype=torch.int64)
    for i in range(vertexCount):
        labels[i] = uf.find(i)

    # Re-index so labels start at 0..(num_clusters-1)
    # e.g. "unique(labels, return_inverse=True)" does exactly that
    labels = torch.unique(labels, return_inverse=True)[1]

    return labels
