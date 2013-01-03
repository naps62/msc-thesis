#include <omp.h>
#include <iostream>
#include <limits>
#include "FVLib.h"
#include "FVL/FVParameters.h"

#include "prof_tottime.h"

//
//	GLOBALS
//
#define PROFILE
struct MyParameters {
	string mesh_file;
	string velocity_file;
	string initial_file;
	string output_file;
	double final_time;
	double anim_time;
	int anim_jump;
	double dirichlet;
	double CFL;

	public:
	// Constructor receives parameter file
	MyParameters(string parameters_filename) {
		FVL::FVParameters para(parameters_filename);
		this->mesh_file		= para.getString("MeshName");
		this->velocity_file	= para.getString("VelocityFile");
		this->initial_file	= para.getString("PoluInitFile");
		this->output_file	= para.getString("OutputFile");
		this->final_time	= para.getDouble("FinalTime");
		this->anim_time		= para.getDouble("AnimTimeStep");
		this->anim_jump		= para.getInteger("NbJump");
		this->dirichlet		= para.getDouble("DirichletCondition");
		this->CFL			= para.getDouble("CFL");
	}
};

#ifdef PROFILE_LIMITED
unsigned mliters;
#endif

int threads;




#include "compute_flux.h"
#include "update.h"





int main(int argc, char *argv[])
{
#ifdef PROFILE
	PROFILE_INIT();
#endif

	string parameter_filename;

	if ( argc > 1 )
		parameter_filename = argv[1];
	else
		parameter_filename = "param.xml";


	threads = omp_get_num_procs();

	//string mesh_filename,velo_filename,pol_filename,pol_ini_filename;
	string name;
	// read the parameter
	MyParameters para(parameter_filename.c_str());
//	mesh_filename=para.getString("MeshName");
//	velo_filename=para.getString("VelocityFile");
//	pol_ini_filename=para.getString("PoluInitFile");
//	string pol_fname = para.getString("PoluFile");

//	double dirichlet = para.getDouble("DirichletCondition");

	double time,final_time,dt,h,S;
//	size_t nbiter,nbjump;
	FVMesh2D m;
	FVCell2D *ptr_c;
	FVEdge2D *ptr_e;
	// read the mesh
	m.read(para.mesh_file.c_str());
	FVVect<double> pol(m.getNbCell()),flux(m.getNbEdge());
	FVVect<FVPoint2D<double> > V(m.getNbCell());
	// read the  data  and initialisation
	FVio velocity_file(para.velocity_file.c_str(),FVREAD);
	velocity_file.get(V,time,name);
	FVio polu_ini_file(para.initial_file.c_str(),FVREAD);
	polu_ini_file.get(pol,time,name);
	final_time=para.final_time;
//	nbjump=para.getInteger("NbJump");
	// compute the Mesh parameter
	h=1.e20;
	m.beginCell();
	while((ptr_c=m.nextCell()))
	{
		S=ptr_c->area;
		ptr_c->beginEdge();
		while((ptr_e=ptr_c->nextEdge()))
		{
			if(h*ptr_e->length>S) h=S/ptr_e->length;
		}
	}





	unsigned   edge_count = m.getNbEdge();
	double     max_vel    = numeric_limits<double>::min();
	double *   fluxes     = new double[ edge_count ];
	double *   lengths    = new double[ edge_count ];
	double *   velocities = new double[ edge_count ];
	unsigned * lefts      = new unsigned[ edge_count ];
	unsigned * rights     = new unsigned[ edge_count ];
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		FVEdge2D *fv_edge = m.getEdge( e );

		fluxes[ e ]   = flux[ e ];
		lengths[ e ]  = fv_edge->length;
		lefts[e]  = fv_edge->leftCell->label - 1;
		rights[e] = ( fv_edge->rightCell )
					  ? fv_edge->rightCell->label - 1
					  : numeric_limits<unsigned>::max();

		double normal[2];
		normal[0] = fv_edge->normal.x;
		normal[1] = fv_edge->normal.y;
		double v_left[2];
		v_left[0] = V[ lefts[e] ].x;
		v_left[1] = V[ lefts[e] ].y;
		double v_right[2];
		if ( rights[e] < numeric_limits<unsigned>::max() )
		{
			v_right[0] = V[ rights[e] ].x;
			v_right[1] = V[ rights[e] ].y;
		}
		else
		{
			v_right[0] = v_left[0];
			v_right[1] = v_left[1];
		}

		velocities[e] = ( v_left[0] + v_right[0] ) * 0.5 * normal[0]
					  + ( v_left[1] + v_right[1] ) * 0.5 * normal[1];

		max_vel = ( abs( velocities[e] ) > max_vel )
				? abs( velocities[e] )
				: max_vel
				;
	}


	dt = h / max_vel;



	unsigned   cell_count      = m.getNbCell();
	unsigned   cell_edge_count = 0;
	//Cell *cells = new Cell[ cell_count ];
	double *   polutions       = new double[ cell_count ];
	double *   areas           = new double[ cell_count ];
	unsigned * indexes         = new unsigned[ cell_count ];
	for ( unsigned c = 0 ; c < cell_count ; ++c )
	{
		FVCell2D *fv_cell = m.getCell( c );

		polutions[c] = pol[c];
		areas[c] = fv_cell->area;
		indexes[c] = cell_edge_count;
		cell_edge_count += fv_cell->nb_edge;
	}




	unsigned   i         = 0;
	unsigned   c         = 0;
	unsigned   cell_last = cell_count - 1;
	unsigned * edges     = new unsigned[ cell_edge_count ];
	while ( c < cell_last )
	{
		FVCell2D *fv_cell = m.getCell( c );
		unsigned e = 0;
		while ( i < indexes[c+1] )
		{
			edges[i] = fv_cell->edge[e]->label - 1;
			++e;
			++i;
		}
		++c;
	}{
		FVCell2D *fv_cell = m.getCell( c );
		unsigned e = 0;
		while ( i < cell_edge_count )
		{
			edges[i] = fv_cell->edge[e]->label - 1;
			++e;
			++i;
		}
	}


	// the main loop
#ifdef PROFILE_LIMITED
	for ( mliters = 0 ; mliters < PROFILE_LIMITED ; ++mliters)
#else
	for ( ; time < final_time ; time += dt )
#endif
	{
		compute_flux(
			polutions,
			velocities,
			lefts,
			rights,
			fluxes,
			para.dirichlet,
			edge_count)
		;
		update(
			polutions,
			areas,
			fluxes,
			lengths,
			indexes,
			edges,
			lefts,
			dt,
			cell_edge_count,
			cell_count)
		;
	}

	for ( unsigned c = 0; c < cell_count ; ++c )
		pol[ c ] = polutions[c];

	FVio pol_file( para.output_file.c_str() ,FVWRITE);
	pol_file.put(pol,time,"polution");

	delete[] fluxes;
	delete[] lengths;
	delete[] velocities;
	delete[] lefts;
	delete[] rights;
	delete[] polutions;
	delete[] areas;
	delete[] indexes;
	delete[] edges;


#ifdef PROFILE
	PROFILE_OUTPUT();
	PROFILE_CLEANUP();
#endif
}
