import torch


def face_normals(vertices: torch.Tensor, faces: torch.Tensor):
    """Returns the face normals, where the vertices are counter clockwise.

    Args:
        vertices (torch.Tensor): The vertices of the mesh of dim (B, V, 3)
        faces (torch.Tensor: The faces which contains the vertices idx, hence the value
            is between 0..V-1 and the dim is (F, 3).

    Returns:
        (torch.Tensor): Returns the normals of the faces of dim (B, F, 3).
    """
    fv = vertices[:, faces]  # (B, F, 3, 3)
    a = fv[:, :, 1] - fv[:, :, 0]
    b = fv[:, :, 2] - fv[:, :, 0]
    f_normals = torch.linalg.cross(a, b, dim=-1)  # (F, 3)
    return f_normals / torch.norm(f_normals, dim=-1).unsqueeze(-1)


def compute_alpha(a, b):
    # https://trimesh.org/trimesh.geometry.html weighted vertex normal
    n = torch.linalg.cross(a, b, dim=-1)  # (F, 3)
    da = torch.norm(a, dim=-1)
    db = torch.norm(b, dim=-1)
    dn = torch.norm(n, dim=-1)
    return torch.arcsin(dn / (da * db))


def normalize(a: torch.Tensor):
    return a / torch.norm(a, dim=-1).unsqueeze(-1)


def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(-1)


def face_angles(vertices: torch.Tensor, faces: torch.Tensor):
    """Calculates the angles of input triangles.

    Args:
        vertices (torch.Tensor): The vertices of the mesh of dim (B, V, 3)
        faces (torch.Tensor: The faces which contains the vertices idx of dim (F, 3).

    Returns:
        (torch.Tensor): Returns the angles of the edges of the faces of dim (B, F, 3).
    """
    B = vertices.shape[0]
    F = faces.shape[0]

    fv = vertices[:, faces]
    u = normalize(fv[:, :, 1] - fv[:, :, 0])
    v = normalize(fv[:, :, 2] - fv[:, :, 0])
    w = normalize(fv[:, :, 2] - fv[:, :, 1])
    result = torch.zeros((B, F, 3), device=vertices.device)
    result[:, :, 0] = torch.arccos(torch.clip(dot(u, v), -1, 1))
    result[:, :, 1] = torch.arccos(torch.clip(dot(-u, w), -1, 1))
    result[:, :, 2] = torch.pi - result[:, :, 0] - result[:, :, 1]
    return result


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor):
    """Calculates the vertex normals of a given mesh.

    Args:
        vertices (torch.Tensor): The vertices of the mesh of dim (B, V, 3)
        faces (torch.Tensor: The faces which contains the vertices idx of dim (F, 3).

    Returns:
        (torch.Tensor): Returns the normals of the vertices of dim (B, V, 3).
    """
    B, V, _ = vertices.shape
    F = faces.shape[0]

    f_angles = face_angles(vertices, faces)  # (B, F, 3)

    # the flat indexes, if you have 2 batches the flat list can be just splitted by the
    # half and then the left half is the first batch.
    f_idx = torch.arange(F).repeat_interleave(3).repeat(B)
    v_idx = faces.reshape(-1).repeat(B)
    b_idx = torch.arange(B).repeat_interleave(F * 3)  # 0,...,0,1...1,...,F*3

    vf = torch.zeros(B, V, F, device=vertices.device)
    vf[b_idx, v_idx, f_idx] = f_angles.reshape(-1)  # (B, V, F)

    f_normals = face_normals(vertices, faces)  # (B, F, 3)
    v_normals = torch.bmm(vf, f_normals)  # (B, V, 3)
    v_normals = v_normals / torch.norm(v_normals, dim=-1).unsqueeze(-1)  # (V, 3)

    return v_normals
