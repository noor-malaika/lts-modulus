# import vtk

# # Read VTP file (PolyData)
# reader = vtk.vtkXMLPolyDataReader()
# reader.SetFileName("graph_9.vtp")
# reader.Update()
# polydata = reader.GetOutput()

# # Convert PolyData to Unstructured Grid
# tri_filter = vtk.vtkTriangleFilter()
# tri_filter.SetInputData(polydata)
# tri_filter.Update()

# # Write out as .vtk (unstructured)
# writer = vtk.vtkUnstructuredGridWriter()
# writer.SetFileName("output.vtk")
# writer.SetInputData(tri_filter.GetOutput())
# writer.Write()
# exit()
import meshio

# Read the converted .vtk file
mesh = meshio.read("uscg_1.vtk")

# Write it to Abaqus .inp format
meshio.write("output.inp", mesh)
