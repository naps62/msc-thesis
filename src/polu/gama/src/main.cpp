/*
 * main.cpp
 *
 *  Created on: December 14, 2012
 *      Author: Miguel Palhas
 */
#define __GAMA_SKEL

// gama
#include <gama.h>

// beast
#include <beast/program_options/option_var.hpp>
using beast::program_options::option_var;
using beast::program_options::parse_option_vars;

// fvl

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

	FVL::GAMAMesh2D mesh(data.mesh_file);
}
