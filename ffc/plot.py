"This module provides functionality for plotting finite elements."

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2010-12-07"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

__all__ = ["plot"]

from numpy import dot, cross, array, sin, cos, pi
from numpy.linalg import norm

from ffc.fiatinterface import create_element
from ffc.log import warning, error, info

# Import Soya3D
try:
    import soya
    from soya.sphere import Sphere
    _soya_imported = True
except:
    _soya_imported = False

# Colors for elements
element_colors = {"Brezzi-Douglas-Marini":    (1.00, 1.00, 0.00),
                  "Crouzeix-Raviart":         (1.00, 0.25, 0.25),
                  "Lagrange":                 (0.00, 1.00, 0.00),
                  "Morley":                   (0.50, 0.50, 0.50),
                  "Nedelec 1st kind H(curl)": (0.90, 0.30, 0.00),
                  "Raviart-Thomas":           (0.90, 0.60, 0.00)}

def plot(element):
    "Plot finite element."

    # Check if Soya3D has been imported
    if not _soya_imported:
        warning("Unable to plot element, Soya3D not available (install package python-soya).")
        return

    # Create cell model
    cell, is3d = create_cell_model(element)

    # Create dof models
    dofs = create_dof_models(element)

    # Render plot window
    render([cell] + dofs, is3d)

def render(models, is3d=True):
    "Render given list of models."

    # Note that we view from the positive z-axis, and not from the
    # negative y-axis. This should make no difference since the
    # element dofs are symmetric anyway and it plays better with
    # the default camera settings in Soya.

    # Initialize Soya
    soya.init("FFC plot", sound=1)

    # Create scene
    scene = soya.World()
    scene.atmosphere = soya.Atmosphere()
    scene.atmosphere.bg_color = (1.0, 1.0, 1.0, 1.0)

    # Define rotation around y-axis
    if is3d:
        class RotatingBody(soya.Body):
            def advance_time(self, proportion):
                self.rotate_y(2.0 * proportion)
    else:
        class RotatingBody(soya.Body):
            def advance_time(self, proportion):
                self.rotate_z(2.0 * proportion)

    # Add all models
    for model in models:
        rotating = RotatingBody(scene, model)

    # Set light
    light = soya.Light(scene)
    if is3d:
        light.set_xyz(1.0, 5.0, 5.0)
    else:
        light.set_xyz(0.0, 0.0, 1.0)
    light.cast_shadow = 1
    light.shadow_color = (0.0, 0.0, 0.0, 0.5)

    # Set camera
    camera = soya.Camera(scene)
    camera.ortho = 0
    camera.set_xyz(0.0, 10, 50.0)
    p = camera.position()
    if is3d:
        camera.fov = 1.8
        p.set_xyz(0.0, 0.4, 0.0)
    else:
        camera.fov = 2.5
        p.set_xyz(0.0, 0.0, 0.0)
    camera.look_at(p)
    soya.set_root_widget(camera)

    # Main loop
    soya.MainLoop(scene).main_loop()

def tangents(n):
    "Return normalized tangent vectors for plane defined by given vector."

    # Figure out which vector to take cross product with
    eps = 1e-10
    e = array((1.0, 0.0, 0.0))
    if norm(cross(n, e)) < eps:
        e = array((0.0, 1.0, 0.0))

    # Take cross products and normalize
    t0 = cross(n, e)
    t0 = t0 / norm(t0)
    t1 = cross(n, t0)
    t1 = t1 / norm(t0)

    return t0, t1

def Cylinder(scene, p0, p1, r, color=(0.0, 0.0, 0.0, 1.0)):
    "Return model for cylinder from p0 to p1 with radius r."

    # Convert to NumPy array
    if isinstance(p0, soya.Vertex):
        p0 = array((p0.x, p0.y, p0.z))
        p1 = array((p1.x, p1.y, p1.z))
    else:
        p0 = array(p0)
        p1 = array(p1)

    # Get tangent vectors for plane
    n = p0 - p1
    n = n / norm(n)
    t0, t1 = tangents(n)

    # Traverse the circles
    num_steps = 10
    dtheta = 2.0*pi / float(num_steps)
    for i in range(num_steps):

        # Compute coordinates for square
        dx0 = cos(i*dtheta)*t0 + sin(i*dtheta)*t1
        dx1 = cos((i + 1)*dtheta)*t0 + sin((i + 1)*dtheta)*t1
        x0 = p0 + r*dx0
        x1 = p0 + r*dx1
        x2 = p1 + r*dx0
        x3 = p1 + r*dx1

        # Cover square by two triangles
        v0 = soya.Vertex(scene, x0[0], x0[1], x0[2], diffuse=color)
        v1 = soya.Vertex(scene, x1[0], x1[1], x1[2], diffuse=color)
        v2 = soya.Vertex(scene, x2[0], x2[1], x2[2], diffuse=color)
        v3 = soya.Vertex(scene, x3[0], x3[1], x3[2], diffuse=color)
        f0 = soya.Face(scene, (v0, v1, v2))
        f1 = soya.Face(scene, (v1, v2, v3))
        f0.double_sided = 1
        f1.double_sided = 1

    # Extract model
    model = scene.to_model()

    return model

def Arrow(scene, x, n, l=0.3):
    "Return model for arrow from x in direction n."

    # Convert to Numpy arrays
    x = array(x)
    n = array(n)

    # Get tangents
    t0, t1 = tangents(n)

    # Dimensions for arrow
    r = 0.04*l
    R1 = 0.1*l
    R2 = 0.2*l

    # Create cylinders
    x0 = x
    x1 = x + l*n
    x2 = x1 - R2*n
    l0 = Cylinder(scene, x0, x1, r)
    l1 = Cylinder(scene, x1 - 0.5*r*n, x1 - 0.5*r*n - R2*n + R1*t1, r)
    l2 = Cylinder(scene, x1 - 0.5*r*n, x1 - 0.5*r*n - R2*n - R1*t1, r)

    # Extract model
    return scene.to_model()

def UnitTetrahedron(color=(0.0, 1.0, 0.0, 0.5)):
    "Return model for unit tetrahedron."

    # Create separate scene (since we will extract a model, not render)
    scene = soya.World()

    # Create vertices
    v0 = soya.Vertex(scene, 0.0, 0.0, 0.0, diffuse=color)
    v1 = soya.Vertex(scene, 1.0, 0.0, 0.0, diffuse=color)
    v2 = soya.Vertex(scene, 0.0, 1.0, 0.0, diffuse=color)
    v3 = soya.Vertex(scene, 0.0, 0.0, 1.0, diffuse=color)

    # Create edges
    e0 = Cylinder(scene, v0, v1, 0.005)
    e0 = Cylinder(scene, v0, v2, 0.005)
    e0 = Cylinder(scene, v0, v3, 0.005)
    e0 = Cylinder(scene, v1, v2, 0.005)
    e0 = Cylinder(scene, v1, v3, 0.005)
    e0 = Cylinder(scene, v2, v3, 0.005)

    # Create faces
    f0 = soya.Face(scene, (v1, v2, v3))
    f1 = soya.Face(scene, (v0, v2, v3))
    f2 = soya.Face(scene, (v0, v1, v3))
    f3 = soya.Face(scene, (v0, v1, v2))

    # Make faces double sided
    f0.double_sided = 1
    f1.double_sided = 1
    f2.double_sided = 1
    f3.double_sided = 1

    # Extract model
    model = scene.to_model()

    return model

def UnitTriangle(color=(0.0, 1.0, 0.0, 0.5)):
    "Return model for unit tetrahedron."

    # Create separate scene (since we will extract a model, not render)
    scene = soya.World()

    # Create vertice
    v0 = soya.Vertex(scene, 0.0, 0.0, 0.0, diffuse=color)
    v1 = soya.Vertex(scene, 1.0, 0.0, 0.0, diffuse=color)
    v2 = soya.Vertex(scene, 0.0, 1.0, 0.0, diffuse=color)

    # Create face
    f = soya.Face(scene, (v0, v1, v2))

    # Make face double sided
    f.double_sided = 1

    # Extract model
    model = scene.to_model()

    return model

def PointEvaluation(x):
    "Return model for point evaluation at given point."

    # Make sure point is 3D
    x = to3d(x)

    # Create separate scene (since we will extract a model, not render)
    scene = soya.World()

    # Define material (color) for the sphere
    material = soya.Material()
    material.diffuse = (0.0, 0.0, 0.0, 1.0)

    # Create sphere
    sphere = Sphere(scene, material=material)

    # Scale and moveand move to coordinate
    sphere.scale(0.05, 0.05, 0.05)
    p = sphere.position()
    p.set_xyz(x[0], x[1], x[2])
    sphere.move(p)

    # Extract model
    model = scene.to_model()

    return model

def DirectionalEvaluation(x, n, flip=False):
    "Return model for directional evaluation at given point in given direction."

    # Make sure points are 3D
    x = to3d(x)
    n = to3d(n)

    # Create separate scene (since we will extract a model, not render)
    scene = soya.World()

    # Normalize
    n = array(n)
    n = 0.75 * n / norm(n)

    # Flip normal if necessary
    if flip and not pointing_outwards(x, n):
        info("Flipping direction of arrow so it points outward.")
        n = -n

    # Create arrow
    arrow = Arrow(scene, x, n)

    # Extract model
    model = scene.to_model()

    return model

def DirectionalDerivative(x, n):
    "Return model for directional derivative at given point in given direction."

    # Make sure points are 3D
    x = to3d(x)
    n = to3d(n)

    # Create separate scene (since we will extract a model, not render)
    scene = soya.World()

    # Normalize
    n = array(n)
    n = 0.75 * n / norm(n)

    # Create line
    line = Cylinder(scene, x - 0.07*n, x + 0.07*n, 0.005)

    # Extract model
    model = scene.to_model()

    return model

def create_cell_model(element):
    "Create Soya3D model for cell."

    # Get color
    family = element.family()
    if not family in element_colors:
        warning("Don't know a good color for elements of type '%s', using default color." % family)
        family = "Lagrange"
    color = element_colors[family]
    color = (color[0], color[1], color[2], 0.7)

    # Create model based on domain type
    domain = element.cell().domain()
    if domain == "triangle":
        return UnitTriangle(color), False
    elif domain == "tetrahedron":
        return UnitTetrahedron(color), True

    error("Unable to plot element, unhandled cell type: %s" % str(domain))

def create_dof_models(element):
    "Create Soya3D models for dofs."

    # Dofs that should be flipped if point in the "wrong" direction
    directional = {"PointScaledNormalEval": True,
                   "PointEdgeTangent":      False,
                   "PointFaceTangent":      False}

    # Create FIAT element and get dofs
    fiat_element = create_element(element)
    dofs = fiat_element.dual_basis()

    # Iterate over dofs and add models
    models = []
    for (i, dof) in enumerate(dofs):

        # Get type of dof
        dof_type = dof.get_type_tag()

        # Create model based on dof type
        L = dof.get_point_dict()
        if dof_type == "PointEval":

            # Point evaluation, just get point
            points = L.keys()
            if not len(points) == 1:
                error("Strange dof, single point expected for point evaluation.")
            x = points[0]

            # Generate model
            models.append(PointEvaluation(x))

        elif dof_type == "PointNormalDeriv":

            # FIXME: Special fix for Morley elements until Rob fixes in FIAT
            if element.family() == "Morley":
                L = morley_fix(L, i)

            print L

            # Evaluation of derivatives at point
            points = L.keys()
            if not len(points) == 1:
                error("Strange dof, single point expected for point evaluation.")
            x = points[0]
            n = [xx[0] for xx in L[x]]

            print x, n

            # Generate model
            models.append(DirectionalDerivative(x, n))

        elif dof_type in directional:

            # Normal evaluation, get point and normal
            points = L.keys()
            if not len(points) == 1:
                error("Strange dof, single point expected for point evaluation.")
            x = points[0]
            n = [xx[0] for xx in L[x]]

            # Generate model
            models.append(DirectionalEvaluation(x, n, directional[dof_type]))

        elif dof_type in ("FrobeniusIntegralMoment", "IntegralMoment", "ComponentPointEval"):

            warning("Not plotting interior moment for now.")

        else:
            error("Unable to plot dof, unhandled dof type: %s" % str(dof_type))

    return models

def pointing_outwards(x, n):
    "Check if n is pointing inwards, used for flipping dofs."
    eps = 1e-10
    x = array(x) + 0.1*array(n)
    return x[0] < -eps or x[1] < -eps or x[2] < -eps or x[2] > 1.0 - x[0] - x[1] + eps

def to3d(x):
    "Make sure point is 3D."
    if len(x) == 2:
        x = (x[0], x[1], 0.0)
    return x

def morley_fix(L, i):
    "Special fix for Morley elements until Rob fixes in FIAT."
    if not len(L) == 0:
        warning("Looks like Rob has fixed the Morley elements in FIAT so this fix is no longer required...")
        return L
    return { 3: {(0.5, 0.0): [( 0.0, (0,)), (-1.0,  (1,))]},
             4: {(0.5, 0.5): [( 1.0, (0,)), ( 1.0,  (1,))]},
             5: {(0.0, 0.5): [(-1.0, (0,)), ( 0.0,  (1,))]} }[i]
