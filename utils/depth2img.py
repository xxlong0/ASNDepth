import numpy as np
import cv2
import pdb


class Depth2Img(object):
    """Convert gray image to rgb representation for plotting

    Args:
        depth
    Returns:
        colored depth in [0,1]
    """

    def __init__(self, max_depth=5.):
        """
        """
        self.max_depth = max_depth

    def __call__(self, depth):
        """
        """
        depth[depth < 0] = 0
        invalid_mask = depth > 254
        
        #normalize to (0, 1)
        # dmax = np.max(depth[~invalid_mask])
        # dmin = np.min(depth[~invalid_mask])
        dmax = self.max_depth
        dmin = 0.
        d = dmax - dmin
        depth = (depth - dmin) / d if dmax!=dmin else 1e5
        
        depth = depth * 255
        depth = np.uint8(depth)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET) 
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB) 
        
        depth_color = depth_color / 255.
        
#         pdb.set_trace()

        for i in range(3):
            depth_color[:, :, i][invalid_mask] = 0

#         depth_color = np.transpose(depth_color, (2, 0, 1))
        return depth_color
