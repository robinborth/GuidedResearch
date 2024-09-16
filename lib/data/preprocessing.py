import cv2
import torch


def extract_mask(depth: torch.Tensor, threshold: float = 0.8):
    return (depth < threshold) & (depth != 0)  # threshold in m


def biliteral_filter(image, dilation, sigma_color, sigma_space):
    img = cv2.bilateralFilter(
        src=image.detach().cpu().numpy(),
        d=dilation,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )
    return torch.tensor(img).to(image)


def point2normal(point: torch.Tensor):
    """Calculate the normal image from the camera image.

    We calculate the normal in camera space, hence we also need to normalize with the
    depth information. Note that the normal is basically on which direction we have the
    stepest decent.

    More can be be found here:
    https://stackoverflow.com/questions/34644101/

    They have (-dz/dx,-dz/dy,1), however we are in camera space hence we need to
    calculate the gradient in pixel space, e.g. also the delta x and delta y are in
    camera space.

    Args:
        depth (torch.Tensor): Camera image of dim (H, W, 3).

    Returns:
        (torch.Tensor): The normal image based on the depth/camera of dim (H, W, 3).
    """
    # in order to calc that only with depth image, we need to make sure that the
    # depth is in pixel space.
    point = point.clone()

    # make sure that on the boundary is nothing wrong calculated
    # point_mask = point.sum(-1) != 0
    # point[~point_mask] = torch.nan  # some large value

    H, W, _ = point.shape
    normals = torch.zeros_like(point)
    normals[..., 2] = -1  # make sure that the default normal looks to the camera

    x_right = torch.arange(2, W)
    x_left = torch.arange(0, W - 2)
    dzx = point[:, x_right, 2] - point[:, x_left, 2]
    dx = point[:, x_right, 0] - point[:, x_left, 0]
    normals[:, 1:-1, 0] = dzx / dx

    y_right = torch.arange(2, H)
    y_left = torch.arange(0, H - 2)
    dzy = point[y_right, :, 2] - point[y_left, :, 2]
    dy = point[y_right, :, 1] - point[y_left, :, 1]
    normals[1:-1, :, 1] = dzy / dy

    # remove artefacs
    normals = torch.nan_to_num(normals, 0)
    normals[:1, :, :] = 0
    normals[-1:, :, :] = 0
    normals[:, :1, :] = 0
    normals[:, -1:, :] = 0

    # create the maks based on the valid normal values
    normal_mask = normals.sum(-1) != 0
    normals[~normal_mask] = 0.0

    # to make them right-hand and follow camera space convention +X right +Y up
    # +Z towards the camera
    normals = -normals

    # normalized between [-1, 1]
    normals = torch.nn.functional.normalize(normals, dim=-1)

    return normals, normal_mask
