from preflight import require

require(gnm=True, packages=("numpy", "trimesh"))

from gnm.shape import gnm_numpy
from gnm.shape import semantic_sampler
import numpy as np
import trimesh # For visualization

# Load the GNM head model.
gnm = gnm_numpy.GNM.from_local(
    version=gnm_numpy.GNMMajorVersion.V3,
    variant=gnm_numpy.GNMVariant.HEAD,
)

# Get the template (average) face mesh.
template_vertices = gnm.template_vertex_positions
faces = gnm.triangles

# Save or visualize the mesh (example using trimesh).
# mesh = trimesh.Trimesh(vertices=template_vertices, faces=faces, process=False)
# mesh.show()
# mesh.export("template_face.obj")



expr_sampler = semantic_sampler.ExpressionSampler()

# Available expression labels.
print(expr_sampler.expression_label_mapping)

# Sample a 'happy' expression.
happy_expression = expr_sampler.sample_expression(
    semantic_sampler.Expression.HAPPY, num_samples=1
)[0]

vertices_happy = gnm(expression=happy_expression)
mesh_happy = trimesh.Trimesh(vertices=vertices_happy, faces=faces)
mesh_happy.show()
# mesh_happy.export("happy_face.obj")
