import numpy as np


# This is the ellipsoid shape model class
class Ellipsoid():
    # Initialize
    def __init__(self, axes, center=np.zeros(3)):
        super().__init__()

        # Ellipsoid axes and center
        self.axes = axes

        # Center of mass position
        self.center = center

    # This method computes if a point is exterior
    def is_exterior(self, pos):
        # Do difference w.r.t. center
        dpos = pos - self.center

        # Compute ellipsoid equation
        f = (dpos[0]/self.axes[0])**2 \
            + (dpos[1]/self.axes[1])**2 \
            + (dpos[2]/self.axes[2])**2

        # Assign is_exterior
        is_exterior = False
        if f >= 0:
            is_exterior = True

        return is_exterior

    # This method computes altitude
    def compute_altitude(self, pos):
        # Compute radius and altitude
        r = np.linalg.norm(pos)
        alt = r - self.axes[0]

        return alt

    # This method computes gravity potential
    def plot3D(self, ax, scale=1, n=100, color='b'):
        # Create a grid of points in spherical coordinates
        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(0, np.pi, n)

        # Parametric equations for the ellipsoid
        x = self.axes[0] * np.outer(np.cos(u), np.sin(v))
        y = self.axes[1] * np.outer(np.sin(u), np.sin(v))
        z = self.axes[2] * np.outer(np.ones_like(u), np.cos(v))

        # Plot surface
        ax.plot_surface(x*scale, y*scale, z*scale, rstride=5, cstride=5,
                        color=color, alpha=0.2, edgecolor='k', zorder=-1)
