import hoomd # 4.8.2
import math
import numpy as np
import itertools
import gsd.hoomd # gsd version 4.0.0
import hoomd.md.integrate
import h5py
import os
import copy
import csv
import shutil

def determine_whether_merge_or_link(frame_, i, parameters):
    frame_merge = relaxed_frame()
    frame_link = relaxed_frame()

    frame_merge.copy_frame_from_snapshot(frame_)
    frame_merge.copy_property(frame_)
    frame_merge.merge(i, parameters)
    energy_density_merge = frame_merge.potential_eng / len(frame_merge.triangles)

    frame_link.copy_frame_from_snapshot(frame_)
    frame_link.copy_property(frame_)
    frame_link.insert_a_wedge(i, parameters)
    energy_density_link = frame_link.potential_eng / len(frame_link.triangles)

    if energy_density_merge < energy_density_link:
        frame_.copy_frame_from_snapshot(frame_merge)
        frame_.copy_property(frame_merge)
        frame_.write_frame_to_gsd()
    else:
        frame_.copy_frame_from_snapshot(frame_link)
        frame_.copy_property(frame_link)
        frame_.write_frame_to_gsd()

class relaxed_frame(gsd.hoomd.Frame):
    def __init__(self):
        super().__init__()

        self.boundary_triangles = []
        self.boundary_open_angles = []
        self.if_at_boundary = []

        self.triangles = []
        self.export_dir = 'a'
        self.nt_i = []
        self.potential_eng = 0.

        self.mu = 0.
        self.muN = 0.
        self.epsilon_hp = 0.
        self.hp_eng = 0.

        self.tot_eng = 0.

    def copy_property(self, my_frame):
        self.boundary_triangles = copy.deepcopy(my_frame.boundary_triangles)
        self.boundary_open_angles = copy.deepcopy(my_frame.boundary_open_angles)
        self.if_at_boundary = copy.deepcopy(my_frame.if_at_boundary)
        
        self.triangles = copy.deepcopy(my_frame.triangles)
        self.export_dir = copy.deepcopy(my_frame.export_dir)
        self.nt_i = copy.deepcopy(my_frame.nt_i)
        self.potential_eng = copy.deepcopy(my_frame.potential_eng)

        self.mu = copy.deepcopy(my_frame.mu)
        self.muN = copy.deepcopy(my_frame.muN)
        self.epsilon_hp = copy.deepcopy(my_frame.epsilon_hp)
        self.hp_eng = copy.deepcopy(my_frame.hp_eng)

        self.tot_eng = copy.deepcopy(my_frame.tot_eng)

    def copy_frame_from_snapshot(self, snapshot):
        self.particles.N = copy.deepcopy(snapshot.particles.N)
        self.particles.position = copy.deepcopy(snapshot.particles.position)
        self.particles.orientation = copy.deepcopy(snapshot.particles.orientation)
        self.particles.typeid = copy.deepcopy(snapshot.particles.typeid)
        self.particles.types = copy.deepcopy(snapshot.particles.types)
        self.particles.velocity = copy.deepcopy(snapshot.particles.velocity)
        self.configuration.box = copy.deepcopy(snapshot.configuration.box)

        self.bonds.N = copy.deepcopy(snapshot.bonds.N)
        self.bonds.typeid = copy.deepcopy(snapshot.bonds.typeid)
        self.bonds.types = copy.deepcopy(snapshot.bonds.types)
        self.bonds.group = copy.deepcopy(snapshot.bonds.group)

        self.dihedrals.N = copy.deepcopy(snapshot.dihedrals.N)
        self.dihedrals.types = copy.deepcopy(snapshot.dihedrals.types)
        self.dihedrals.typeid = copy.deepcopy(snapshot.dihedrals.typeid)
        self.dihedrals.group = copy.deepcopy(snapshot.dihedrals.group)

    def update_one_boundary_open_angle(self, boundary_index):
        boundary_triangles = self.boundary_triangles
        selected_point = boundary_triangles[boundary_index][0]
        neighbor1 = boundary_triangles[boundary_index - 1][0]
        neighbor2 = boundary_triangles[(boundary_index + 1) % len(boundary_triangles)][0]
        
        vec1 = np.array(self.particles.position[neighbor1]) - np.array(self.particles.position[selected_point])
        vec2 = np.array(self.particles.position[neighbor2]) - np.array(self.particles.position[selected_point])
        alpha = np.arccos(round(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), 4))
        if self.nt_i[selected_point] > 2:
            self.boundary_open_angles[boundary_index] = alpha
        else:
            self.boundary_open_angles[boundary_index] = 2 * np.pi - alpha

    def update_all_boundary_open_angles(self):
        boundary_points_ = (np.array(self.boundary_triangles))[:, 0].tolist()
        num_of_boundary_points = len(boundary_points_)
        for index in np.arange(num_of_boundary_points):
            self.update_one_boundary_open_angle(index)
    
    def initialize_dataset(self):
        exportdir = self.export_dir

        if not os.path.exists(exportdir):
            os.mkdir(exportdir)
        else:
            shutil.rmtree(exportdir)
            os.mkdir(exportdir)

        if not os.path.isfile(exportdir + '/data.hdf5'):
            with h5py.File(exportdir + '/data.hdf5', "a") as f:
                f.create_dataset("potential_energy", (10000,))
                f.create_dataset("trimer_number", (10000,))
                f.create_dataset("particle_number", (10000,))
                f.create_dataset("bond_number", (10000,))
                f.create_dataset("dihedral_number", (10000,))
                f.create_dataset("hydrophobic_eng", (10000,))
                f.create_dataset("muN", (10000,))
                f.create_dataset("tot_eng", (10000,))

    def write_data_to_h5py(self, i):
        with h5py.File(self.export_dir + '/data.hdf5', "a") as f:
            f["potential_energy"][i] = self.potential_eng
            f["trimer_number"][i] = len(self.triangles)
            f["particle_number"][i] = self.particles.N
            f["bond_number"][i] = self.bonds.N
            f["dihedral_number"][i] = self.dihedrals.N
            f["hydrophobic_eng"][i] = self.hp_eng
            f["muN"][i] = self.muN
            f["tot_eng"][i] = self.tot_eng
            f.close()

    def write_frame_to_gsd(self):
        with gsd.hoomd.open(name=self.export_dir + '/frame.gsd', mode='a') as f:
            f.append(self)
            f.flush()
            f.close()
    
    def relax_frame(self, parameters):
        simulation = hoomd.Simulation(device=parameters['my_device'])

        theta_0 = 2 * np.arcsin(parameters['r_0'] / np.sqrt(12 * (parameters['R_0'])**2 - 3 * (parameters['r_0'])**2))

        harmonic = hoomd.md.bond.Harmonic()
        harmonic.params['A-A'] = dict(k=parameters['k_s'], r0=parameters['r_0'])
        dihedral = hoomd.md.dihedral.Periodic()
        dihedral.params['A-A-A-A'] = dict(k=parameters['k_d'], d=-1, n=1, phi0=np.pi - theta_0)

        constant_volume = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        fire = hoomd.md.minimize.FIRE(
            dt=0.001, force_tol=1e-5, angmom_tol=1e-5, energy_tol=1e-6, forces=[harmonic, dihedral], methods=[constant_volume]
        )
        simulation.create_state_from_snapshot(self)
        simulation.operations.integrator = fire
        simulation.run(0)

        thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
            filter=hoomd.filter.All()
        )
        simulation.operations.computes.append(thermodynamic_properties)
        while not fire.converged:
            simulation.run(parameters['single_run_time'])
        self.copy_frame_from_snapshot(simulation.state.get_snapshot())

        self.potential_eng = thermodynamic_properties.potential_energy

    def initialize_capsid(self, parameters_):
        self.configuration.box = [parameters_['box_L'], parameters_['box_L'], parameters_['box_L'], 0, 0, 0]
        
        N_particles = 3
        self.particles.N = N_particles
        self.particles.position = [[0, 0.5, 0], [0, -0.5, 0], [np.sqrt(3)/2, 0, 0]]
        self.particles.orientation = [(1, 0, 0, 0)] * N_particles
        self.particles.types = ['Hex', 'Pen', 'Bou']
        self.particles.typeid = [2] * N_particles

        N_bonds = 3
        self.bonds.N = N_bonds
        self.bonds.types = ['A-A']
        self.bonds.typeid = [0] * N_bonds
        self.bonds.group = [[0, 1], [1, 2], [2, 0]]

        N_dihedrals = 0
        self.dihedrals.N = N_dihedrals
        self.dihedrals.types = ['A-A-A-A']
        self.dihedrals.typeid = []
        self.dihedrals.group = []

        self.boundary_triangles = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        self.if_at_boundary = [1, 1, 1]

        self.triangles = [[0, 1, 2]]
        self.nt_i = [1, 1, 1]

        self.boundary_open_angles = [5 * np.pi / 3, 5 * np.pi / 3, 5 * np.pi / 3]

        self.export_dir = os.getcwd() + '/box_L_' + str(parameters_['box_L']) + '_k_s_' + str(parameters_['k_s']) + '_r_0_' + str(parameters_['r_0']) + '_k_d_' + str(parameters_['k_d']) + '_R_0_' + str(parameters_['R_0']) + '_mu_' + str(parameters_['mu']) + '_epsilon_hp_' + str(parameters_['epsilon_hp']) + '_single_run_time_' + str(parameters_['single_run_time'])

        self.mu = parameters_['mu']
        self.muN = parameters_['mu']
        self.epsilon_hp = parameters_['epsilon_hp']
        self.hp_eng = 0.

        self.tot_eng = - self.muN

        self.initialize_dataset()
        self.relax_frame(parameters=parameters_)
        
        self.write_data_to_h5py(i=0)
        self.write_frame_to_gsd()

        with open(self.export_dir + '/parameters.csv', 'w', newline='') as csvfile:
            parameter_writer = csv.writer(csvfile, delimiter=' ')
            parameter_writer.writerows(parameters_)

    def add_a_free_trimer(self, boundary_bond_index_at_new_trimer, parameters_):
        selected_triangle = self.boundary_triangles[boundary_bond_index_at_new_trimer]
        selected_bond = [selected_triangle[0], selected_triangle[1]]
        selected_bond_length = np.linalg.norm(self.particles.position[selected_bond[0]] - self.particles.position[selected_bond[1]])
        bond_vec = self.particles.position[selected_triangle[1]] - self.particles.position[selected_triangle[0]]
        surface_vec = self.particles.position[selected_triangle[2]] - self.particles.position[selected_triangle[1]]
        normal_vec = np.cross(bond_vec, surface_vec)
        bond_mid = 0.5 * (self.particles.position[selected_triangle[0]] + self.particles.position[selected_triangle[1]])
        outer_vec = np.cross(bond_vec, normal_vec)
        distance_new_particle_to_bond_mid = np.sqrt((parameters_['r_0'])**2 - selected_bond_length**2 / 4.)
        theta_0 = 2 * np.arcsin(parameters_['r_0'] / np.sqrt(12 * (parameters_['R_0'])**2 - 3 * (parameters_['r_0'])**2))
        
        new_particle_pos = bond_mid\
            + distance_new_particle_to_bond_mid * np.cos(theta_0) * outer_vec / np.linalg.norm(outer_vec)\
            - distance_new_particle_to_bond_mid * np.sin(theta_0) * normal_vec / np.linalg.norm(normal_vec)
        new_particle_index = self.particles.N

        self.particles.N = self.particles.N + 1
        self.particles.position = np.append(self.particles.position, np.array([new_particle_pos.tolist()]), axis=0)
        self.particles.orientation = np.append(self.particles.orientation, [[1, 0, 0, 0]], axis=0)
        self.particles.typeid = np.append(self.particles.typeid, 2)
        self.particles.velocity = np.append(self.particles.velocity, [(self.particles.velocity[selected_triangle[0]] + self.particles.velocity[selected_triangle[1]]) / 2], axis=0)

        self.bonds.N = self.bonds.N + 2
        self.bonds.typeid = np.append(self.bonds.typeid, 0)
        self.bonds.typeid = np.append(self.bonds.typeid, 0)
        self.bonds.group = np.append(self.bonds.group, [[selected_triangle[0], new_particle_index]], axis=0)
        self.bonds.group = np.append(self.bonds.group, [[new_particle_index, selected_triangle[1]]], axis=0)

        self.dihedrals.N = self.dihedrals.N + 1
        self.dihedrals.typeid = np.append(self.dihedrals.typeid, 0)
        self.dihedrals.group = np.append(self.dihedrals.group, [[selected_triangle[-1], selected_triangle[-2], selected_triangle[-3], new_particle_index]], axis=0)

        self.boundary_triangles.insert(boundary_bond_index_at_new_trimer, [selected_triangle[0], new_particle_index, selected_triangle[1]])
        self.boundary_triangles.insert(boundary_bond_index_at_new_trimer + 1, [new_particle_index, selected_triangle[1], selected_triangle[0]])
        self.boundary_triangles.pop(boundary_bond_index_at_new_trimer + 2)

        self.if_at_boundary.append(1)

        self.hp_eng = self.hp_eng + 2 * self.epsilon_hp * (self.nt_i[selected_triangle[0]] + self.nt_i[selected_triangle[1]])

        self.nt_i[selected_triangle[0]] = self.nt_i[selected_triangle[0]] + 1
        self.nt_i[selected_triangle[1]] = self.nt_i[selected_triangle[1]] + 1
        self.nt_i.append(1)

        self.muN = self.muN + self.mu

        self.tot_eng = self.potential_eng + self.hp_eng - self.muN

        self.boundary_open_angles.insert(boundary_bond_index_at_new_trimer + 1, 0)
        self.update_one_boundary_open_angle(boundary_index=boundary_bond_index_at_new_trimer)
        self.update_one_boundary_open_angle(boundary_index=boundary_bond_index_at_new_trimer + 1)
        self.update_one_boundary_open_angle((boundary_bond_index_at_new_trimer + 2) % len(self.boundary_open_angles))

        self.triangles.append([selected_triangle[0], new_particle_index, selected_triangle[1]])

    def insert_a_wedge(self, vertex_index_at_new_wedge, parameters_):
        left_triangle = self.boundary_triangles[vertex_index_at_new_wedge - 1]
        right_triangle = self.boundary_triangles[vertex_index_at_new_wedge]
        seleted_point = right_triangle[0]

        self.bonds.N = self.bonds.N + 1
        self.bonds.typeid = np.append(self.bonds.typeid, 0)
        self.bonds.group = np.append(self.bonds.group, [[left_triangle[0], right_triangle[1]]], axis=0)

        self.dihedrals.N = self.dihedrals.N + 2
        self.dihedrals.typeid = np.append(self.dihedrals.typeid, 0)
        self.dihedrals.typeid = np.append(self.dihedrals.typeid, 0)
        self.dihedrals.group = np.append(
            self.dihedrals.group,
            [[left_triangle[-1], left_triangle[-2], left_triangle[-3], right_triangle[1]]],
            axis=0
            )
        self.dihedrals.group = np.append(
            self.dihedrals.group,
            [[left_triangle[0], right_triangle[0], right_triangle[1], right_triangle[2]]],
            axis=0
            )
        
        self.relax_frame(parameters=parameters_)

        self.boundary_triangles.insert(vertex_index_at_new_wedge, [left_triangle[0], right_triangle[1], right_triangle[0]])
        self.boundary_triangles.pop(vertex_index_at_new_wedge + 1)
        self.boundary_triangles.pop(vertex_index_at_new_wedge - 1)

        self.if_at_boundary[seleted_point] = 0

        self.triangles.append([left_triangle[0], right_triangle[1], seleted_point])

        self.hp_eng = self.hp_eng + 2 * self.epsilon_hp * (self.nt_i[left_triangle[0]] + self.nt_i[seleted_point] + self.nt_i[right_triangle[1]])

        self.muN = self.muN + self.mu

        self.tot_eng = self.potential_eng + self.hp_eng - self.muN

        self.nt_i[left_triangle[0]] = self.nt_i[left_triangle[0]] + 1
        self.nt_i[seleted_point] = self.nt_i[seleted_point] + 1
        self.nt_i[right_triangle[1]] = self.nt_i[right_triangle[1]] + 1

        self.boundary_open_angles.pop(vertex_index_at_new_wedge)
        self.update_all_boundary_open_angles()

        if self.nt_i[seleted_point] == 6:
            self.particles.typeid[seleted_point] = 0
        elif self.nt_i[seleted_point] == 5:
            self.particles.typeid[seleted_point] = 1
        if self.nt_i[left_triangle[0]] == 6:
            self.particles.typeid[left_triangle[0]] = 0
        elif self.nt_i[left_triangle[0]] == 5:
            self.particles.typeid[left_triangle[0]] = 1
        if self.nt_i[right_triangle[1]] == 6:
            self.particles.typeid[right_triangle[1]] = 0
        elif self.nt_i[right_triangle[1]] == 5:
            self.particles.typeid[right_triangle[1]] = 1

    def merge(self, merge_index, parameters_):
        left_triangle = self.boundary_triangles[merge_index - 1]
        right_triangle = self.boundary_triangles[merge_index]
        left_point = left_triangle[0]
        selected_point = right_triangle[0]
        right_point = right_triangle[1]

        self.dihedrals.N = self.dihedrals.N + 1
        self.dihedrals.typeid = np.append(self.dihedrals.typeid, 0)
        self.dihedrals.group = np.append(
            self.dihedrals.group, 
            [[left_triangle[-1], right_triangle[0], right_triangle[1], right_triangle[2]]],
            axis=0
            )

        if right_point > left_point:
            for i, dihedral_group in enumerate(self.dihedrals.group):
                for j, point_index in enumerate(dihedral_group):
                    if point_index == left_point:
                        self.dihedrals.group[i][j] = right_point - 1
                    if point_index > left_point:
                        self.dihedrals.group[i][j] = self.dihedrals.group[i][j] - 1
            for i, bond_group in enumerate(self.bonds.group):
                if all([left_point, selected_point] == bond_group) or all([selected_point, left_point] == bond_group):
                    bond_index_to_be_deleted = i
                for j, point_index in enumerate(bond_group):
                    if point_index == left_point:
                        self.bonds.group[i][j] = right_point - 1
                    if point_index > left_point:
                        self.bonds.group[i][j] = self.bonds.group[i][j] - 1
            for i, triangle_group in enumerate(self.triangles):
                for j, point_index in enumerate(triangle_group):
                    if point_index == left_point:
                        self.triangles[i][j] = right_point - 1
                    if point_index > left_point:
                        self.triangles[i][j] = self.triangles[i][j] - 1
        else:
            for i, dihedral_group in enumerate(self.dihedrals.group):
                for j, point_index in enumerate(dihedral_group):
                    if point_index == left_point:
                        self.dihedrals.group[i][j] = right_point
                    if point_index > left_point:
                        self.dihedrals.group[i][j] = self.dihedrals.group[i][j] - 1
            for i, bond_group in enumerate(self.bonds.group):
                if all([left_point, selected_point] == bond_group) or all([selected_point, left_point] == bond_group):
                    bond_index_to_be_deleted = i
                for j, point_index in enumerate(bond_group):
                    if point_index == left_point:
                        self.bonds.group[i][j] = right_point
                    if point_index > left_point:
                        self.bonds.group[i][j] = self.bonds.group[i][j] - 1
            for i, triangle_group in enumerate(self.triangles):
                for j, point_index in enumerate(triangle_group):
                    if point_index == left_point:
                        self.triangles[i][j] = right_point
                    if point_index > left_point:
                        self.triangles[i][j] = self.triangles[i][j] - 1
        
        self.bonds.N = self.bonds.N - 1

        self.bonds.group = np.delete(self.bonds.group, bond_index_to_be_deleted, 0)
        self.bonds.typeid = np.delete(self.bonds.typeid, bond_index_to_be_deleted, 0)

        self.boundary_triangles[merge_index - 2][-2] = right_point
        self.boundary_triangles.pop(merge_index)
        self.boundary_triangles.pop(merge_index - 1)

        self.if_at_boundary[selected_point] = 0
        self.if_at_boundary.pop(left_point)

        for i, triangle in enumerate(self.boundary_triangles):
            for j, point_index in enumerate(triangle):
                if point_index > left_point:
                    self.boundary_triangles[i][j] = self.boundary_triangles[i][j] - 1

        self.hp_eng = self.hp_eng + 2 * self.epsilon_hp * self.nt_i[right_point] * self.nt_i[left_point]
        
        self.nt_i[right_point] = self.nt_i[right_point] + self.nt_i[left_point]
        if self.nt_i[right_point] == 5:
            self.particles.typeid[right_point] = 1
        elif self.nt_i[right_point] == 6:
            self.particles.typeid[right_point] = 0
        if self.nt_i[selected_point] == 5:
            self.particles.typeid[selected_point] = 1
        elif self.nt_i[selected_point] == 6:
            self.particles.typeid[selected_point] = 0
        self.nt_i.pop(left_point)

        self.particles.N = self.particles.N - 1
        self.particles.position = np.delete(self.particles.position, left_point, axis=0)
        self.particles.orientation = np.delete(self.particles.orientation, left_point, axis=0)
        self.particles.typeid = np.delete(self.particles.typeid, left_point, axis=0)
        self.particles.velocity = np.delete(self.particles.velocity, left_point, axis=0)

        self.relax_frame(parameters=parameters_)

        self.tot_eng = self.potential_eng + self.hp_eng - self.muN

        self.boundary_open_angles.pop(merge_index)
        self.boundary_open_angles.pop(merge_index - 1)
        self.update_all_boundary_open_angles()

    def remove_a_free_trimer(self, index_of_removed_particle):
        right_triangle = self.boundary_triangles[index_of_removed_particle]
        right_bond = [right_triangle[0], right_triangle[1]]
        selected_point = right_triangle[0]
        left_triangle = self.boundary_triangles[index_of_removed_particle - 1]
        left_bond = [left_triangle[0], left_triangle[1]]

        self.particles.N = self.particles.N - 1
        self.particles.position = np.delete(self.particles.position, selected_point, axis=0)
        self.particles.orientation = np.delete(self.particles.orientation, selected_point, axis=0)
        self.particles.typeid = np.delete(self.particles.typeid, selected_point, axis=0)
        self.particles.velocity = np.delete(self.particles.velocity, selected_point, axis=0)

        bond_index_to_be_deleted = []
        for i, bond_group in enumerate(self.bonds.group):
            for j, point_index in enumerate(bond_group):
                if point_index == selected_point:
                    bond_index_to_be_deleted.append(i)
                if point_index > selected_point:
                    self.bonds.group[i][j] = self.bonds.group[i][j] - 1
        bond_index_to_be_deleted.sort(reverse=True)
        for i in bond_index_to_be_deleted:
            self.bonds.group = np.delete(self.bonds.group, i, 0)
            self.bonds.typeid = np.delete(self.bonds.typeid, i, 0)
        self.bonds.N = self.bonds.N - 2

        for i, dihedral_group in enumerate(self.dihedrals.group):
            for j, point_index in enumerate(dihedral_group):
                if point_index == selected_point:
                    dihedral_index_to_be_deleted = i
                if point_index > selected_point:
                    self.dihedrals.group[i][j] = self.dihedrals.group[i][j] - 1
        deleted_dihedral_group = self.dihedrals.group[dihedral_index_to_be_deleted].tolist()
        self.dihedrals.group = np.delete(self.dihedrals.group, dihedral_index_to_be_deleted, 0)
        self.dihedrals.typeid = np.delete(self.dihedrals.typeid, dihedral_index_to_be_deleted, 0)
        self.dihedrals.N = self.dihedrals.N - 1

        triangle_index_to_be_deleted = []
        for i, triangle_group in enumerate(self.triangles):
            for j, point_index in enumerate(triangle_group):
                if point_index == selected_point:
                    triangle_index_to_be_deleted.append(i)
                    if j == 0:
                        next_point_index_in_the_boundary_triangle = 3
                    else:
                        next_point_index_in_the_boundary_triangle = 0
                if point_index > selected_point:
                    self.triangles[i][j] = self.triangles[i][j] - 1
        self.triangles.pop(triangle_index_to_be_deleted[0])

        self.boundary_triangles.insert(index_of_removed_particle, [left_bond[0], right_bond[1], deleted_dihedral_group[next_point_index_in_the_boundary_triangle]])
        self.boundary_triangles.pop(index_of_removed_particle + 1)
        self.boundary_triangles.pop(index_of_removed_particle - 1)

        i = 0
        for triangle in self.boundary_triangles:
            for j, point_index in enumerate(triangle):
                if point_index > selected_point:
                    self.boundary_triangles[i][j] = self.boundary_triangles[i][j] - 1
            i = i + 1
        
        self.if_at_boundary.pop(index_of_removed_particle)

        self.boundary_open_angles.pop(index_of_removed_particle)
        self.update_one_boundary_open_angle(index_of_removed_particle - 1)
        self.update_one_boundary_open_angle(index_of_removed_particle % len(self.boundary_open_angles))

        self.hp_eng = self.hp_eng - 2 * self.epsilon_hp * (self.nt_i[left_bond[0]] + self.nt_i[right_bond[1]] - 2)

        self.muN = self.muN - self.mu

        self.tot_eng = self.potential_eng + self.hp_eng - self.muN

        self.nt_i[left_bond[0]] = self.nt_i[left_bond[0]] - 1
        self.nt_i[right_bond[1]] = self.nt_i[right_bond[1]] - 1
        if self.nt_i[left_bond[0]] == 5:
            self.particles.typeid[left_bond[0]] = 1
        elif self.nt_i[left_bond[0]] == 6:
            self.particles.typeid[left_bond[0]] = 0
        if self.nt_i[right_bond[1]] == 5:
            self.particles.typeid[right_bond[1]] = 1
        elif self.nt_i[right_bond[1]] == 6:
            self.particles.typeid[right_bond[1]] = 0
        self.nt_i.pop(selected_point)

    def remove_a_wedge(self, index_of_removed_bond, parameters_):
        selected_triangle = self.boundary_triangles[index_of_removed_bond]
        selected_bond = [selected_triangle[0], selected_triangle[1]]

        for i, bond_group in enumerate(self.bonds.group):
            if ((selected_bond[0] in bond_group) and (selected_bond[1] in bond_group)):
                bond_index_to_be_deleted = i
        self.bonds.N = self.bonds.N - 1
        self.bonds.group = np.delete(self.bonds.group, bond_index_to_be_deleted, 0)
        self.bonds.typeid = np.delete(self.bonds.typeid, bond_index_to_be_deleted, 0)

        dihedral_index_to_be_deleted = []
        for i, diehdral_group in enumerate(self.dihedrals.group):
            if ((selected_bond[0] in diehdral_group) and (selected_bond[1] in diehdral_group)):
                dihedral_index_to_be_deleted.append(i)
        dihedral_index_to_be_deleted.sort(reverse=True)
        deleted_dihedral_group0 = self.dihedrals.group[dihedral_index_to_be_deleted[0]].tolist()
        deleted_dihedral_group1 = self.dihedrals.group[dihedral_index_to_be_deleted[1]].tolist()
        for i in dihedral_index_to_be_deleted:
            self.dihedrals.group = np.delete(self.dihedrals.group, i, 0)
            self.dihedrals.typeid = np.delete(self.dihedrals.typeid, i, 0)
        self.dihedrals.N = self.dihedrals.N - 2

        if deleted_dihedral_group0[-1] == selected_triangle[1]:
            self.boundary_triangles[index_of_removed_bond] = [deleted_dihedral_group0[2], deleted_dihedral_group0[1], deleted_dihedral_group0[0]]
            if deleted_dihedral_group1[0] == selected_triangle[0]:
                self.boundary_triangles.insert(index_of_removed_bond + 1, [deleted_dihedral_group1[1], deleted_dihedral_group1[2], deleted_dihedral_group1[3]])
            else:
                self.boundary_triangles.insert(index_of_removed_bond + 1, [deleted_dihedral_group1[2], deleted_dihedral_group1[1], deleted_dihedral_group1[0]])
        elif deleted_dihedral_group0[0] == selected_triangle[1]:
            self.boundary_triangles[index_of_removed_bond] = [deleted_dihedral_group0[1], deleted_dihedral_group0[2], deleted_dihedral_group0[3]]
            if deleted_dihedral_group1[0] == selected_triangle[0]:
                self.boundary_triangles.insert(index_of_removed_bond + 1, [deleted_dihedral_group1[1], deleted_dihedral_group1[2], deleted_dihedral_group1[3]])
            else:
                self.boundary_triangles.insert(index_of_removed_bond + 1, [deleted_dihedral_group1[2], deleted_dihedral_group1[1], deleted_dihedral_group1[0]])
        elif deleted_dihedral_group0[0] == selected_triangle[0]:
            self.boundary_triangles.insert(index_of_removed_bond + 1, [deleted_dihedral_group0[1], deleted_dihedral_group0[2], deleted_dihedral_group0[3]])
            if deleted_dihedral_group1[-1] == selected_triangle[1]:
                self.boundary_triangles[index_of_removed_bond] = [deleted_dihedral_group1[2], deleted_dihedral_group1[1], deleted_dihedral_group1[0]]
            else:
                self.boundary_triangles[index_of_removed_bond] = [deleted_dihedral_group1[1], deleted_dihedral_group1[2], deleted_dihedral_group1[3]]
        elif deleted_dihedral_group0[-1] == selected_triangle[0]:
            self.boundary_triangles.insert(index_of_removed_bond + 1, [deleted_dihedral_group0[2], deleted_dihedral_group0[1], deleted_dihedral_group0[0]])
            if deleted_dihedral_group1[-1] == selected_triangle[1]:
                self.boundary_triangles[index_of_removed_bond] = [deleted_dihedral_group1[2], deleted_dihedral_group1[1], deleted_dihedral_group1[0]]
            else:
                self.boundary_triangles[index_of_removed_bond] = [deleted_dihedral_group1[1], deleted_dihedral_group1[2], deleted_dihedral_group1[3]]

        for i, triangle_group in enumerate(self.triangles):
            if ((selected_bond[0] in triangle_group) and (selected_bond[1] in triangle_group)):
                triangle_index_to_be_deleted = i
        self.triangles.pop(triangle_index_to_be_deleted)

        self.particles.typeid[selected_triangle[0]] = 2
        self.particles.typeid[selected_triangle[1]] = 2
        self.particles.typeid[selected_triangle[2]] = 2

        self.relax_frame(parameters=parameters_)

        self.if_at_boundary[selected_triangle[2]] = 1

        self.hp_eng = self.hp_eng - 2 * self.epsilon_hp * (self.nt_i[selected_triangle[0]] + self.nt_i[selected_triangle[1]] + self.nt_i[selected_triangle[2]] - 3)

        self.muN = self.muN - self.mu

        self.tot_eng = self.potential_eng + self.hp_eng - self.muN

        self.nt_i[selected_triangle[0]] = self.nt_i[selected_triangle[0]] - 1
        self.nt_i[selected_triangle[1]] = self.nt_i[selected_triangle[1]] - 1
        self.nt_i[selected_triangle[2]] = self.nt_i[selected_triangle[2]] - 1

        self.boundary_open_angles.insert(index_of_removed_bond + 1, 0)
        self.update_all_boundary_open_angles()

# Three methods below are for inverse merge.
# For a potential inversible merge point at the boundary,
# first collect all bonds, dihedeals, triangles and inner points that is connected by a bond.
# Depending on the specific inverse merge bond, we collect all related bonds, dihedrals and triangles
# on one side of the to-be-open bond (left or right), including the bond itself.
# Then we add a new particle, edit the corresponding point index in the collected set,
# remove the dihedral at the to-be-open bond, and add another bond from the ineer to the new point.
    def fond_related_bonds_dihedrals_triangles_innerpoints(self, index_of_boundary_points):
        selected_point = self.boundary_triangles[index_of_boundary_points][0]
        
        bond_index_list = []
        dihedral_index_list = []
        triangle_index_list = []
        bond_innerpoint_list = []

        for i, bond_group in enumerate(self.bonds.group):
            for point_index in bond_group:
                if point_index == selected_point:
                    bond_index_list.append(i)
                    if point_index == bond_group[0]:
                        if self.if_at_boundary[bond_group[1]] == 0:
                            bond_innerpoint_list.append(bond_group[1].tolist())
                    elif point_index == bond_group[1]:
                        if self.if_at_boundary[bond_group[0]] == 0:
                            bond_innerpoint_list.append(bond_group[0].tolist())
        for i, dihedral_group in enumerate(self.dihedrals.group):
            if selected_point in dihedral_group:
                dihedral_index_list.append(i)
        for i, triangle_group in enumerate(self.triangles):
            if selected_point in triangle_group:
                triangle_index_list.append(i)

        return bond_index_list, dihedral_index_list, triangle_index_list, bond_innerpoint_list

    def collect_objects_from_one_side(self, index_of_boundary_point, index_of_inner_point, bond_index_list, dihedral_index_list, triangle_index_list, bond_innerpoint_list):
        selected_point = self.boundary_triangles[index_of_boundary_point][0]
        inner_point = bond_innerpoint_list[index_of_inner_point]

        collected_bond = []
        collected_dihedral = []
        collected_triangle = []

        loop_point = self.boundary_triangles[index_of_boundary_point - 1][0]
        while((loop_point != inner_point)):
            for i, bond_index in enumerate(bond_index_list):
                if loop_point in self.bonds.group[bond_index]:
                    collected_bond.append(bond_index)
                    bond_index_list.pop(i)
                    break

            index_to_be_deleted = []
            for i, dihedral_index in enumerate(dihedral_index_list):
                if loop_point in self.dihedrals.group[dihedral_index]:
                    collected_dihedral.append(dihedral_index)
                    index_to_be_deleted.append(i)
            index_to_be_deleted.sort(reverse=True)
            for i in index_to_be_deleted:
                dihedral_index_list.pop(i)

            for index in reversed(collected_dihedral):
                this_dihedral = self.dihedrals.group[index].tolist()
                if (inner_point == this_dihedral[1]) and (selected_point == this_dihedral[2]):
                    left_inner_point = this_dihedral[0]
                    right_inner_point = this_dihedral[3]
                    dihedral_tobe_deleted = index
                    break
                elif (inner_point == this_dihedral[2]) and (selected_point == this_dihedral[1]):
                    left_inner_point = this_dihedral[3]
                    right_inner_point = this_dihedral[0]
                    dihedral_tobe_deleted = index
                    break
            
            for i, triangle_index in enumerate(triangle_index_list):
                this_triangle = self.triangles[triangle_index]
                if (loop_point in this_triangle):
                    collected_triangle.append(triangle_index)
                    for new_point in this_triangle:
                        if new_point not in [selected_point, loop_point]:
                            loop_point = new_point
                            break
                    triangle_index_list.pop(i)

        return collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted
    
    def inverse_merge(self, index_of_boundary_point, index_of_inner_point, bond_innerpoint_list, parameters_, collected_bond, collected_dihedral, collected_triangle, left_inner_point, right_inner_point, dihedral_tobe_deleted):
        selected_point = self.boundary_triangles[index_of_boundary_point][0]
        inner_point = bond_innerpoint_list[index_of_inner_point]

        self.particles.N = self.particles.N + 1
        self.particles.position = np.append(self.particles.position, np.array([(self.particles.position[selected_point]).tolist()]), axis=0)
        self.particles.orientation = np.append(self.particles.orientation, [[1, 0, 0, 0]], axis=0)
        self.particles.typeid = np.append(self.particles.typeid, 2)
        self.particles.typeid[inner_point] = 2
        self.particles.velocity = np.append(self.particles.velocity, [self.particles.velocity[selected_point]], axis=0)

        new_point = self.particles.N - 1

        for bond_index in collected_bond:
            for j, point in enumerate((self.bonds.group[bond_index]).tolist()):
                if point == selected_point:
                    self.bonds.group[bond_index][j] = new_point
        self.bonds.N = self.bonds.N + 1
        self.bonds.typeid = np.append(self.bonds.typeid, 0)
        self.bonds.group = np.append(self.bonds.group, [[inner_point, new_point]], axis=0)

        for dihedral_index in collected_dihedral:
            for j, point in enumerate((self.dihedrals.group[dihedral_index]).tolist()):
                if point == selected_point:
                    self.dihedrals.group[dihedral_index][j] = new_point
        self.dihedrals.group = np.delete(self.dihedrals.group, dihedral_tobe_deleted, 0)
        self.dihedrals.typeid = np.delete(self.dihedrals.typeid, dihedral_tobe_deleted, 0)
        self.dihedrals.N = self.dihedrals.N - 1

        for triangle_index in collected_triangle:
            for j, point in enumerate(self.triangles[triangle_index]):
                if point == selected_point:
                    self.triangles[triangle_index][j] = new_point
        
        self.relax_frame(parameters=parameters_)

        self.boundary_triangles[index_of_boundary_point - 1][1] = new_point
        self.boundary_triangles.insert(index_of_boundary_point, [inner_point, selected_point, right_inner_point])
        self.boundary_triangles.insert(index_of_boundary_point, [new_point, inner_point, left_inner_point])

        self.if_at_boundary[inner_point] = 1
        self.if_at_boundary.append(1)

        new_nt_i = len(collected_triangle)
        self.hp_eng = self.hp_eng + 2 * self.epsilon_hp * new_nt_i * (new_nt_i - self.nt_i[selected_point])

        self.tot_eng = self.potential_eng + self.hp_eng - self.muN

        self.nt_i[selected_point] = self.nt_i[selected_point] - new_nt_i
        self.nt_i.append(new_nt_i)

        self.boundary_open_angles.append(0)
        self.boundary_open_angles.append(0)
        self.update_all_boundary_open_angles()

    def close_capsid(self, parameters_):
        self.dihedrals.N = self.dihedrals.N + 3

        for i, selected_triangle in enumerate(self.boundary_triangles):
            self.dihedrals.typeid = np.append(self.dihedrals.typeid, 0)
            self.dihedrals.group = np.append(
                self.dihedrals.group, 
                [[selected_triangle[-1], selected_triangle[-2], selected_triangle[-3], self.boundary_triangles[(i + 2) % 3][0]]],
                axis=0
                )
            
            self.hp_eng = 2 * self.epsilon_hp * self.nt_i[selected_triangle[0]]

            self.nt_i[selected_triangle[0]] = self.nt_i[selected_triangle[0]] + 1

            if self.nt_i[selected_triangle[0]] == 5:
                self.particles.typeid[selected_triangle[0]] = 1
            elif self.nt_i[selected_triangle[0]] == 6:
                self.particles.typeid[selected_triangle[0]] = 0

            self.if_at_boundary[selected_triangle[0]] = 0
        
        self.relax_frame(parameters=parameters_)

        self.muN = self.muN + self.mu

        self.triangles.append([self.boundary_triangles[-1][0], self.boundary_triangles[-2][0], self.boundary_triangles[-3][0]])

        self.tot_eng = self.potential_eng + self.hp_eng - self.muN

        self.boundary_open_angles.clear()
        
        self.boundary_triangles.clear()