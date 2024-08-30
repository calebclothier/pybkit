import numpy as np


# class Vector3D(np.ndarray):
    
#     def __new__(cls, x, y, z):
#         obj = np.asarray([x, y, z]).view(cls)
#         obj.x, obj.y, obj.z = x, y, z
#         return obj

#     def __array_finalize__(self, obj):
#         if obj is None: return
#         self.x = getattr(obj, 'x', None)
#         self.y = getattr(obj, 'y', None)
#         self.z = getattr(obj, 'z', None)

#     def __repr__(self):
#         return f"Vector3D(x={self.x}, y={self.y}, z={self.z})"
    
    
class Vector3D(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __array__(self, dtype=None):
        if dtype:
            return np.array([self.x, self.y, self.z], dtype=dtype)
        else:
            return np.array([self.x, self.y, self.z])
        
    def __repr__(self):
        return f"Vector3D(x={self.x}, y={self.y}, z={self.z})"