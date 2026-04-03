from virus import kmc_frame

import numpy as np
import hoomd # 5.3.1
import random

def read_parameters(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            # Handle my_device separately
            if key == 'my_device':
                if value.upper() == 'CPU':
                    parameters[key] = hoomd.device.CPU()
                elif value.upper() == 'GPU':
                    parameters[key] = hoomd.device.GPU()
            else:
                parameters[key] = float(value) if '.' in value else int(value)
    return parameters
parameters = read_parameters('parameters.txt')

add_a_free_trimer_angle = np.pi / 2
insert_a_wedge_angle = np.pi / 4
merge_angle = np.pi / 8

max_kmc_moves = 1000

# initialize
capsid = kmc_frame.relaxed_frame()
capsid.initialize_capsid(parameters_=parameters)

ite = 0
while (ite < 3000):
    # kmc in the frame
    num_of_boundary_points = len(capsid.boundary_triangles)

    operation_index = capsid.boundary_open_angles.index(min(capsid.boundary_open_angles))
    least_boundary_open_angle = capsid.boundary_open_angles[operation_index]

    if least_boundary_open_angle >= add_a_free_trimer_angle:
        capsid.add_a_free_trimer(operation_index, parameters)
        capsid.write_frame_to_gsd()
        capsid.write_data_to_h5py(i=ite)

    elif insert_a_wedge_angle <= least_boundary_open_angle < add_a_free_trimer_angle:
        capsid.insert_a_wedge(operation_index, parameters)
        capsid.write_frame_to_gsd()
        capsid.write_data_to_h5py(i=ite)
    
    elif merge_angle <= least_boundary_open_angle < insert_a_wedge_angle:
        kmc_frame.determine_whether_merge_or_link(capsid, operation_index, parameters)

    elif least_boundary_open_angle < merge_angle:
        capsid.merge(operation_index, parameters)
        capsid.write_frame_to_gsd()
        capsid.write_data_to_h5py(i=ite)

    if (len(capsid.triangles) > 1) and (len(capsid.boundary_triangles) <= 3):
        capsid.close_capsid(parameters)
        capsid.write_frame_to_gsd()
        capsid.write_data_to_h5py(i=ite + 1)
        break

    else:
        ite = ite + 1
        if ite % 10 == 0 :
            print(ite)

exit()
for i in np.arange(len(capsid.boundary_triangles)):
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# 2 tri
capsid.add_a_free_trimer(0, parameters_=parameters)
print([len(capsid.triangles), capsid.boundary_triangles])
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles)):
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# 3 tri
capsid.add_a_free_trimer(0, parameters_=parameters)
print([len(capsid.triangles), capsid.boundary_triangles])
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles)):
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# insert a wedge to a quadrangle
capsid.insert_a_wedge(0, parameters_=parameters)
print([len(capsid.triangles), capsid.boundary_triangles])
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles))[::-1]:
    capsid.remove_a_wedge(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.insert_a_wedge(i+1, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    bond_index_list, dihedral_index_list, triangle_index_list, inner_point_index_list = capsid.fond_related_bonds_dihedrals_triangles_innerpoints(i)
    collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted = capsid.collect_objects_from_one_side(i, inner_point_index_list[0], bond_index_list=bond_index_list, dihedral_index_list=dihedral_index_list, triangle_index_list=triangle_index_list, bond_innerpoint_list=inner_point_index_list)
    capsid.inverse_merge(index_of_boundary_point=i, index_of_inner_point=inner_point_index_list[0], bond_innerpoint_list=inner_point_index_list, parameters_=parameters, collected_bond=collected_bond, collected_dihedral=collected_dihedral, collected_triangle=collected_triangle, left_inner_point=left_inner_point, right_inner_point=right_inner_point, dihedral_tobe_deleted=dihedral_tobe_deleted)
    capsid.write_frame_to_gsd()

    capsid.merge(i+1, parameters_=parameters)
    capsid.write_frame_to_gsd()

    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# 4 tri to 3 tri through remove-a-wedge
capsid.remove_a_wedge(0, parameters_=parameters)
print([len(capsid.triangles), capsid.boundary_triangles])
capsid.write_frame_to_gsd()

# 3 tri to 4 tri 
capsid.add_a_free_trimer(0, parameters_=parameters)
print([len(capsid.triangles), capsid.boundary_triangles])
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles))[::-1]:
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# merge to form a quadrangle
capsid.merge(2, parameters_=parameters)
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles)):
    capsid.remove_a_wedge(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.insert_a_wedge(i+1, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    bond_index_list, dihedral_index_list, triangle_index_list, inner_point_index_list = capsid.fond_related_bonds_dihedrals_triangles_innerpoints(i)
    collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted = capsid.collect_objects_from_one_side(i, inner_point_index_list[0], bond_index_list=bond_index_list, dihedral_index_list=dihedral_index_list, triangle_index_list=triangle_index_list, bond_innerpoint_list=inner_point_index_list)
    capsid.inverse_merge(index_of_boundary_point=i, index_of_inner_point=inner_point_index_list[0], bond_innerpoint_list=inner_point_index_list, parameters_=parameters, collected_bond=collected_bond, collected_dihedral=collected_dihedral, collected_triangle=collected_triangle, left_inner_point=left_inner_point, right_inner_point=right_inner_point, dihedral_tobe_deleted=dihedral_tobe_deleted)
    capsid.write_frame_to_gsd()

    capsid.merge(i+1, parameters_=parameters)
    capsid.write_frame_to_gsd()

# dismerge from a quadrangle
bond_index_list, dihedral_index_list, triangle_index_list, inner_point_index_list = capsid.fond_related_bonds_dihedrals_triangles_innerpoints(0)
collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted = capsid.collect_objects_from_one_side(0, inner_point_index_list[0], bond_index_list=bond_index_list, dihedral_index_list=dihedral_index_list, triangle_index_list=triangle_index_list, bond_innerpoint_list=inner_point_index_list)
capsid.inverse_merge(index_of_boundary_point=0, index_of_inner_point=inner_point_index_list[0], bond_innerpoint_list=inner_point_index_list, parameters_=parameters, collected_bond=collected_bond, collected_dihedral=collected_dihedral, collected_triangle=collected_triangle, left_inner_point=left_inner_point, right_inner_point=right_inner_point, dihedral_tobe_deleted=dihedral_tobe_deleted)
capsid.write_frame_to_gsd()

# insert a wedge to form a pentagon
capsid.insert_a_wedge(1, parameters_=parameters)
print([len(capsid.triangles), capsid.boundary_triangles])
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles)):
    bond_index_list, dihedral_index_list, triangle_index_list, inner_point_index_list = capsid.fond_related_bonds_dihedrals_triangles_innerpoints(i)
    collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted = capsid.collect_objects_from_one_side(i, inner_point_index_list[0], bond_index_list=bond_index_list, dihedral_index_list=dihedral_index_list, triangle_index_list=triangle_index_list, bond_innerpoint_list=inner_point_index_list)
    capsid.inverse_merge(index_of_boundary_point=i, index_of_inner_point=inner_point_index_list[0], bond_innerpoint_list=inner_point_index_list, parameters_=parameters, collected_bond=collected_bond, collected_dihedral=collected_dihedral, collected_triangle=collected_triangle, left_inner_point=left_inner_point, right_inner_point=right_inner_point, dihedral_tobe_deleted=dihedral_tobe_deleted)
    capsid.write_frame_to_gsd()

    capsid.merge(i+1, parameters_=parameters)
    capsid.write_frame_to_gsd()

    capsid.remove_a_wedge(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.insert_a_wedge(i+1, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# dismerge from a pentagon
bond_index_list, dihedral_index_list, triangle_index_list, inner_point_index_list = capsid.fond_related_bonds_dihedrals_triangles_innerpoints(0)
collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted = capsid.collect_objects_from_one_side(0, inner_point_index_list[0], bond_index_list=bond_index_list, dihedral_index_list=dihedral_index_list, triangle_index_list=triangle_index_list, bond_innerpoint_list=inner_point_index_list)
capsid.inverse_merge(index_of_boundary_point=0, index_of_inner_point=inner_point_index_list[0], bond_innerpoint_list=inner_point_index_list, parameters_=parameters, collected_bond=collected_bond, collected_dihedral=collected_dihedral, collected_triangle=collected_triangle, left_inner_point=left_inner_point, right_inner_point=right_inner_point, dihedral_tobe_deleted=dihedral_tobe_deleted)
capsid.write_frame_to_gsd()

# merge to a pentagon
capsid.merge(1, parameters_=parameters)
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles)):
    bond_index_list, dihedral_index_list, triangle_index_list, inner_point_index_list = capsid.fond_related_bonds_dihedrals_triangles_innerpoints(i)
    collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted = capsid.collect_objects_from_one_side(i, inner_point_index_list[0], bond_index_list=bond_index_list, dihedral_index_list=dihedral_index_list, triangle_index_list=triangle_index_list, bond_innerpoint_list=inner_point_index_list)
    capsid.inverse_merge(index_of_boundary_point=i, index_of_inner_point=inner_point_index_list[0], bond_innerpoint_list=inner_point_index_list, parameters_=parameters, collected_bond=collected_bond, collected_dihedral=collected_dihedral, collected_triangle=collected_triangle, left_inner_point=left_inner_point, right_inner_point=right_inner_point, dihedral_tobe_deleted=dihedral_tobe_deleted)
    capsid.write_frame_to_gsd()

    capsid.merge(i+1, parameters_=parameters)
    capsid.write_frame_to_gsd()

    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_wedge(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.insert_a_wedge(i+1, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# remove a wedge from a pentagon
capsid.remove_a_wedge(4, parameters_=parameters)
print([len(capsid.triangles), capsid.boundary_triangles])
capsid.write_frame_to_gsd()

# 4 tri to 5 tri
capsid.add_a_free_trimer(5, parameters_=parameters)
print([len(capsid.triangles), capsid.boundary_triangles])
capsid.write_frame_to_gsd()

# form a hexagon through insert_a_wedge
capsid.insert_a_wedge(5, parameters_=parameters)
print([len(capsid.triangles), capsid.boundary_triangles])
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles)):
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()
    
    bond_index_list, dihedral_index_list, triangle_index_list, inner_point_index_list = capsid.fond_related_bonds_dihedrals_triangles_innerpoints(i)
    collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted = capsid.collect_objects_from_one_side(i, inner_point_index_list[0], bond_index_list=bond_index_list, dihedral_index_list=dihedral_index_list, triangle_index_list=triangle_index_list, bond_innerpoint_list=inner_point_index_list)
    capsid.inverse_merge(index_of_boundary_point=i, index_of_inner_point=inner_point_index_list[0], bond_innerpoint_list=inner_point_index_list, parameters_=parameters, collected_bond=collected_bond, collected_dihedral=collected_dihedral, collected_triangle=collected_triangle, left_inner_point=left_inner_point, right_inner_point=right_inner_point, dihedral_tobe_deleted=dihedral_tobe_deleted)
    capsid.write_frame_to_gsd()

    capsid.merge(i+1, parameters_=parameters)
    capsid.write_frame_to_gsd()

    capsid.remove_a_wedge(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.insert_a_wedge(i+1, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# remove a wedge from hexagon
capsid.remove_a_wedge(4, parameters_=parameters)
capsid.write_frame_to_gsd()

# merge to a pentagon
capsid.merge(5, parameters_=parameters)
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles))[::-1]:
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()
    
    capsid.remove_a_wedge(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.insert_a_wedge(i+1, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    bond_index_list, dihedral_index_list, triangle_index_list, inner_point_index_list = capsid.fond_related_bonds_dihedrals_triangles_innerpoints(i)
    collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted = capsid.collect_objects_from_one_side(i, inner_point_index_list[0], bond_index_list=bond_index_list, dihedral_index_list=dihedral_index_list, triangle_index_list=triangle_index_list, bond_innerpoint_list=inner_point_index_list)
    capsid.inverse_merge(index_of_boundary_point=i, index_of_inner_point=inner_point_index_list[0], bond_innerpoint_list=inner_point_index_list, parameters_=parameters, collected_bond=collected_bond, collected_dihedral=collected_dihedral, collected_triangle=collected_triangle, left_inner_point=left_inner_point, right_inner_point=right_inner_point, dihedral_tobe_deleted=dihedral_tobe_deleted)
    capsid.write_frame_to_gsd()

    capsid.merge(i+1, parameters_=parameters)
    capsid.write_frame_to_gsd()

# dismerge from a pentagon
bond_index_list, dihedral_index_list, triangle_index_list, inner_point_index_list = capsid.fond_related_bonds_dihedrals_triangles_innerpoints(2)
collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted = capsid.collect_objects_from_one_side(2, inner_point_index_list[0], bond_index_list=bond_index_list, dihedral_index_list=dihedral_index_list, triangle_index_list=triangle_index_list, bond_innerpoint_list=inner_point_index_list)
capsid.inverse_merge(index_of_boundary_point=2, index_of_inner_point=inner_point_index_list[0], bond_innerpoint_list=inner_point_index_list, parameters_=parameters, collected_bond=collected_bond, collected_dihedral=collected_dihedral, collected_triangle=collected_triangle, left_inner_point=left_inner_point, right_inner_point=right_inner_point, dihedral_tobe_deleted=dihedral_tobe_deleted)
capsid.write_frame_to_gsd()

# 5 tri to 4 tri
capsid.remove_a_free_trimer(2)
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles))[::-1]:
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# merge to a quadrangle
capsid.merge(2, parameters_=parameters)
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles)):
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()
    
    capsid.remove_a_wedge(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.insert_a_wedge(i+1, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    bond_index_list, dihedral_index_list, triangle_index_list, inner_point_index_list = capsid.fond_related_bonds_dihedrals_triangles_innerpoints(i)
    collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted = capsid.collect_objects_from_one_side(i, inner_point_index_list[0], bond_index_list=bond_index_list, dihedral_index_list=dihedral_index_list, triangle_index_list=triangle_index_list, bond_innerpoint_list=inner_point_index_list)
    capsid.inverse_merge(index_of_boundary_point=i, index_of_inner_point=inner_point_index_list[0], bond_innerpoint_list=inner_point_index_list, parameters_=parameters, collected_bond=collected_bond, collected_dihedral=collected_dihedral, collected_triangle=collected_triangle, left_inner_point=left_inner_point, right_inner_point=right_inner_point, dihedral_tobe_deleted=dihedral_tobe_deleted)
    capsid.write_frame_to_gsd()

    capsid.merge(i+1, parameters_=parameters)
    capsid.write_frame_to_gsd()

# remove a wedge from quadrangle
capsid.remove_a_wedge(3, parameters_=parameters)
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles)):
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# insert a wedge to form a quadrangle
capsid.insert_a_wedge(4, parameters_=parameters)
print([len(capsid.triangles), capsid.boundary_triangles])
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles))[::-1]:
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()
    
    bond_index_list, dihedral_index_list, triangle_index_list, inner_point_index_list = capsid.fond_related_bonds_dihedrals_triangles_innerpoints(i)
    collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted = capsid.collect_objects_from_one_side(i, inner_point_index_list[0], bond_index_list=bond_index_list, dihedral_index_list=dihedral_index_list, triangle_index_list=triangle_index_list, bond_innerpoint_list=inner_point_index_list)
    capsid.inverse_merge(index_of_boundary_point=i, index_of_inner_point=inner_point_index_list[0], bond_innerpoint_list=inner_point_index_list, parameters_=parameters, collected_bond=collected_bond, collected_dihedral=collected_dihedral, collected_triangle=collected_triangle, left_inner_point=left_inner_point, right_inner_point=right_inner_point, dihedral_tobe_deleted=dihedral_tobe_deleted)
    capsid.write_frame_to_gsd()

    capsid.merge(i+1, parameters_=parameters)
    capsid.write_frame_to_gsd()

    capsid.remove_a_wedge(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.insert_a_wedge(i+1, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# dismerge from a quadrangle
bond_index_list, dihedral_index_list, triangle_index_list, inner_point_index_list = capsid.fond_related_bonds_dihedrals_triangles_innerpoints(3)
collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted = capsid.collect_objects_from_one_side(3, inner_point_index_list[0], bond_index_list=bond_index_list, dihedral_index_list=dihedral_index_list, triangle_index_list=triangle_index_list, bond_innerpoint_list=inner_point_index_list)
capsid.inverse_merge(index_of_boundary_point=3, index_of_inner_point=inner_point_index_list[0], bond_innerpoint_list=inner_point_index_list, parameters_=parameters, collected_bond=collected_bond, collected_dihedral=collected_dihedral, collected_triangle=collected_triangle, left_inner_point=left_inner_point, right_inner_point=right_inner_point, dihedral_tobe_deleted=dihedral_tobe_deleted)
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles)):
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# 4 tri to 3 tri
capsid.remove_a_free_trimer(5)
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles))[::-1]:
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# 3 tri to 2 tri
capsid.remove_a_free_trimer(0)
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles)):
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

# 2 tri to 1 tri
capsid.remove_a_free_trimer(1)
capsid.write_frame_to_gsd()

for i in np.arange(len(capsid.boundary_triangles))[::-1]:
    capsid.add_a_free_trimer(i, parameters_=parameters)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()

    capsid.remove_a_free_trimer(i + 1)
    print([len(capsid.triangles), capsid.boundary_triangles])
    capsid.write_frame_to_gsd()