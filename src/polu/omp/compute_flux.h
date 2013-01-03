/*
 * compute_flux.h
 *
 *  Created on: Jan 3, 2013
 *      Author: naps62
 */

#ifndef COMPUTE_FLUX_H_
#define COMPUTE_FLUX_H_

void
compute_flux
	(
	double *   polutions,
	double *   velocities,
	unsigned * lefts,
	unsigned * rights,
	double *   fluxes,
	double     dirichlet,
	unsigned   edge_count
	)
{

#ifdef PROFILE
	#ifdef PROFILE_WARMUP
	if ( mliters > PROFILE_WARMUP )
	#endif
		PROFILE_START();
#endif

	#pragma omp parallel for num_threads(threads)
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		double polution_left = polutions[ lefts[e] ];
		double polution_right
			= ( rights[e] < numeric_limits<unsigned>::max() )
			? polutions[ rights[e] ]
			: dirichlet
			;
		fluxes[e] = ( velocities[e] < 0 )
		          ? velocities[e] * polution_right
				  : velocities[e] * polution_left
				  ;
	}

#ifdef PROFILE
	#ifdef PROFILE_WARMUP
	if ( mliters > PROFILE_WARMUP )
	{
	#endif
		PROFILE_STOP();
		PROFILE_RETRIEVE_CF();
	#ifdef PROFILE_WARMUP
	}
	#endif
#endif

}

#endif /* COMPUTE_FLUX_H_ */
