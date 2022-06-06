"""
We treat the visible disc as a circle with dimensionless radius 1.
The point on the observed planetary disc is specified by three input parameters.
    1) phase: orbital phase, which increases from 0 degree at primary transit to
    180 degree at secondary eclipse.
    2) rho: fraction radius, which is the projected distance between the point
    and the centre of the disc, Takes value between 0 and 1 inclusive.
    3) alpha: argument of the point, measured anticlockwise from 3 o'clock.

We use the following coordinate system, in a frame centred at the planetary centre.
Consider the primary transit. At this moment, the orbital phase angle is 0.
WLOG assume the orbit is anticlockwise, and we define the following:
    x-axis points towards 3 o'clock.
    y-axis points towards the star.
    z-axis points towards the north pole.
    theta is measured clockwiseky from the z-axis.
    phi is measured anticlockwise from x-axis.
As the star orbiting around the planet in our frame, we fix the axes so that
they are not rotating. Equivalently, to an intertial observer, the centre of
our frame moves with the centre of the planet, but the axes do not rotate.

Since the planet is tidally locked, the substellar and antistellar points are
stationary without respect to the planet. We define the planetocentric longtitude
to be 180E at the substellar point and 0E at the antistellar point. East is in
the direction of the planet' self rotation.

Orbital phase (stellar phase angle) increases from 0 at primary transit
to 180 at secondary eclipse.

Planetocentric longitude is 180E at the substellar point and 0E at the antistellar
point. East is in the direction of the planet's self-rotation.

# We treat the visible disc as a circle with dimensionless radius 1.
# The point on the observed planetary disc is specified by three input parameters.
#     1) phase: orbital phase, which increases from 0 degree at primary transit to
#     180 degree at secondary eclipse.
#     2) rho: fraction radius, which is the projected distance between the point
#     and the centre of the disc, Takes value between 0 and 1 inclusive.
#     3) alpha: argument of the point, measured anticlockwise from 3 o'clock.

# We use the following coordinate system, in a frame centred at the planetary centre.
# Consider the primary transit. At this moment, the orbital phase angle is 0.
# WLOG assume the orbit is anticlockwise, and we define the following:
#     x-axis points towards 3 o'clock.
#     y-axis points towards the star.
#     z-axis points towards the north pole.
#     theta is measured clockwiseky from the z-axis.
#     phi is measured anticlockwise from x-axis.
# As the star orbiting around the planet in our frame, we fix the axes so that
# they are not rotating. Equivalently, to an intertial observer, the centre of
# our frame moves with the centre of the planet, but the axes do not rotate.

"""