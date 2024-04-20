# https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/projection-stage.html

import numpy as np
import torch


def rasterize_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    image_size: int = 256,
    device="cpu",
):
    """A simple function to rasterize a mesh to the sceen.

    We assume that the vertices are in camera space already, hence we only need to
    perform the perspective projection (perspecitve divide), without transforming the
    vertices.

    Args
        vertices (torch.Tensor): The vertices of dimension (Vx3)
        faces (torch.Tensor): The indexes of the vertices of dimension (Fx3)
        image_size (int): The image size, resulting in a canvas of dim (NxN)

    Returns:
        pix_to_face (torch.Tensor): The index of the faces per (NxN)
        bary_coords: The barycentric coordinates for each pixel (NxNx3)
        zbuf (torch.Tensor): The z-value for each pixel (NxN)
    """
    f = 512
    near = 1.0
    N = image_size
    V = vertices.clone().to(device)
    F = faces.clone().to(device)

    # loop over all the vertices in the mesh
    for P_camera in V:
        # convert the vertices into the sceen space
        # note we projected point's z-ccordinate to the inverse of the original point's
        # z-coordinate, which is negative, handling positive z-coordinates will simplity
        # matters later on.
        P_screen = P_camera.clone()
        P_screen[0] = (near * P_camera[0]) / (-P_camera[2])  # x in screen
        P_screen[1] = (near * P_camera[1]) / (-P_camera[2])  # y in screen
        P_screen[2] = -P_camera[2]  # vertex z-coordinate in camera space

        # remap the screen space to NDC space
        # the coordinates in screen space are in [l < x < r] and [b < y < t]
        P_ndc = P_screen.clone()
        P_ndc[0] = ...  # (do we need that?)
        # should be just go with the definition of the cv2 lecture, they do not use NDC

    canvas = torch.randint(0, 256, (N, N, 3), dtype=torch.uint8)
    return canvas


if __name__ == "__main__":

    # defines the vertices of the mesh, note that we have 2 faces
    vertices = torch.tensor(
        np.array(
            [
                [-1, 1, -2],
                [0.5, 1.5, -2],
                [0.1, -1, -3],
                [1.3, 0.2, -2.5],
            ]
        ),
        dtype=torch.float32,
    )

    # TODO define here the attributes, per vertex

    # defines the faces of the mesh
    faces = torch.tensor(
        np.array(
            [
                [2, 1, 0],
                [3, 1, 2],
            ]
        ),
        dtype=torch.int64,
    )

    # TODO return per pixel (N, N, 3) the barycentric_coordinates
    # and the (N, N) indexes of the faces
    image = rasterize_mesh(
        vertices=vertices,
        faces=faces,
    )

    # save the image
    from PIL import Image

    path = "temp/mesh.png"
    img = Image.fromarray(image.detach().cpu().numpy().astype(np.uint8))
    img.save(path)
