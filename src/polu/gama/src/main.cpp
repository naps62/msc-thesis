/*
 * main.cpp
 *
 *  Created on: December 14, 2012
 *      Author: Miguel Palhas
 */

// gama
#include <gama.h>

// beast
#include <beast/program_options/option_var.hpp>
using beast::program_options::option_var;
using beast::program_options::parse_option_vars;

// fvl
#include "FVL/CFVMesh2D.h"
#include "GAMAFVMesh2D.h"
#include "FVL/CFVLib.h"

// std
#include <string>
using std::string;


//
// Parameter structure
//
struct Params {
	option_var<string> mesh_file;
	option_var<string> out_file;
	option_var<string> velocity_file;
	option_var<string> ini_file;
	option_var<string> potential_file;
	option_var<int>    factor_norm;
	option_var<int>    max_concentration;
	option_var<double> dirichlet;
	option_var<int>    final_time;
	option_var<double> anim_step;

	Params()
	: mesh_file     ("mesh",      "mesh.xml",              "Mesh filename"),
	  out_file      ("output",    "polution.xml",          "Output filename"),
	  velocity_file ("velocity",  "velocity.xml",          "Velocity filename"),
	  ini_file      ("ini",       "concentration_ini.xml", "Initial polution file"),
	  potential_file("potential", "v_potential.xml",       "Potential filename"),
	  factor_norm   ("factor",               1,   "Factor Norm"),
	  max_concentration("max_concentration", 1,   "Max concentration"),
	  dirichlet     ("dirichlet",            1.0, "Dirichlet condition"),
	  final_time    ("final",                10,  "Final time"),
	  anim_step     ("anim_step",            1,   "Animation time step")
	  { }
};

void compute_edge_velocities(FVL::GAMAFVMesh2D& mesh, FVL::CFVPoints2D<double>& velocities, smartPtr<double>& vs, double &v_max) {
	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		unsigned int left	= mesh.edge_left_cells[i];
		unsigned int right	= mesh.edge_right_cells[i];

		if (right == NO_RIGHT_CELL)
			right = left;

		double v	= ((velocities.x[left] + velocities.x[right]) * 0.5 * mesh.edge_normals_x[i])
					+ ((velocities.y[left] + velocities.y[right]) * 0.5 * mesh.edge_normals_y[i]);

		vs[i] = v;

		if (abs(v) > v_max || i == 0) {
			v_max = abs(v);
		}
	}
}

double compute_mesh_parameter(FVL::GAMAFVMesh2D& mesh) {
	double h;
	double S;


//	for(unsigned int edge = 0; edge < mesh.num_edges; ++edge) {
//		cout << mesh.edge_lengths[edge] << endl;
//	}

	h = 1.e20;
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		S = mesh.cell_areas[cell];

		for(unsigned int e = 0; e < mesh.cell_edges_count[cell]; ++e) {
			double edge   = mesh.cell_edges[e][cell];
			double length = mesh.edge_lengths[edge];
			if (h * length > S)
				h = S / length;
		}
	}

	return h;
}

void compute_length_area_ratio(FVL::GAMAFVMesh2D &mesh, smartPtr<double>* length_area_ratio) {
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {

		unsigned int edge_limit = mesh.cell_edges_count[cell];
		for(unsigned int edge_i = 0; edge_i < edge_limit; ++edge_i) {
			unsigned int edge = mesh.cell_edges[edge_i][cell];

			length_area_ratio[edge_i][cell] = mesh.edge_lengths[edge] / mesh.cell_areas[cell];
		}
	}
}

// gama memory system
MemorySystem* LowLevelMemAllocator::_memSys = NULL;

//
// main func
//
int main(int argc, char **argv) {
	Params params;
	parse_option_vars(argc, argv);

	RuntimeScheduler* rs = new RuntimeScheduler();

	int i = 0;
	double h, t, dt, v_max = 0;

	// load the mesh
	FVL::GAMAFVMesh2D mesh(params.mesh_file());

	// temporary vectors, before copying to smartPtr's
	FVL::CFVArray<double>    old_polution(mesh.num_cells);		// polution arrays
	FVL::CFVPoints2D<double> old_velocities(mesh.num_cells);	// velocities by cell (to calc vs array)

	// read other input files
	FVL::FVXMLReader velocity_reader(params.velocity_file());
	FVL::FVXMLReader polu_ini_reader(params.ini_file());
	string name;
	velocity_reader.getPoints2D(old_velocities, t, name);
	polu_ini_reader.getVec(old_polution, t, name);
	polu_ini_reader.close();
	velocity_reader.close();

	// smartPtr data
	smartPtr<double> polution(sizeof(double) * mesh.num_cells);
	smartPtr<double> flux(sizeof(double) * mesh.num_edges);
	smartPtr<double> vs(sizeof(double) * mesh.num_edges);
	smartPtr<double> length_area_ratio[3];
	for(int i = 0; i < 3; ++i) {
		length_area_ratio[i] = smartPtr<double>(sizeof(double) * mesh.num_cells);
	}

	// copy old data to new structs
	for(unsigned cell = 0; cell < mesh.num_cells; ++cell) {
		polution[cell] = old_polution[cell];
	}

	// compute velocity vector
	compute_edge_velocities(mesh, old_velocities, vs, v_max);
	h = compute_mesh_parameter(mesh);
	compute_length_area_ratio(mesh, length_area_ratio);


	FVL::FVXMLWriter polution_writer(params.out_file());
//	polution_writer.append(polution, mesh.num_cells, t, "polution");

	dt	= h / v_max;

	// loop control vars
	bool   finished       = false;
	while(!finished) {
		cout << "time: " << t << "   iteration: " << i << "\r";

		if (t + dt > params.final_time()) {
			cout << endl << "Final iteration, adjusting dt" << endl;
			dt = params.final_time() - t;
			finished = true;
		}

		ComputeFlux* compute_flux = new ComputeFlux(mesh, vs, flux, polution, params.dirichlet());
		rs->synchronize();
		rs->submit(compute_flux);
		rs->synchronize();
		Update* update = new Update(mesh, flux, polution, dt);
		rs->submit(update);
		rs->synchronize();

		t += dt;
		++i;
	}

	polution_writer.append(polution, mesh.num_cells, t, "polution");

	polution_writer.save();
	polution_writer.close();
	delete rs;
}
