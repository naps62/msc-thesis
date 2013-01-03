/*
 * update.h
 *
 *  Created on: Jan 3, 2013
 *      Author: naps62
 */

#ifndef UPDATE_H_
#define UPDATE_H_

void
update
	(
	double *   polutions,
	double *   areas,
	double *   fluxes,
	double *   lengths,
	unsigned * indexes,
	unsigned * edges,
	unsigned * lefts,
	double     dt,
	unsigned   index_count,
	unsigned   cell_count
	)
{

#ifdef PROFILE
	#ifdef PROFILE_WARMUP
	if ( mliters > PROFILE_WARMUP )
	#endif
		PROFILE_START();
#endif

	unsigned cell_last = cell_count - 1;

	#pragma omp parallel for num_threads(threads)
	for ( unsigned c = 0 ; c < cell_count ; ++c )
	{
		double cdp = 0;
		unsigned i_limit
			= ( c < cell_last )
			? indexes[c+1]
			: index_count
			;
		for ( unsigned i = indexes[c] ; i < i_limit ; ++i )
		{
			unsigned e = edges[i];
			double edp = dt * fluxes[e] * lengths[e] / areas[c];
			if ( lefts[e] == c )
				cdp -= edp;
			else
				cdp += edp;
		}

		polutions[c] += cdp;
	}

#ifdef PROFILE
	#ifdef PROFILE_WARMUP
	if ( mliters > PROFILE_WARMUP )
	{
	#endif
		PROFILE_STOP();
		PROFILE_RETRIEVE_UP();
	#ifdef PROFILE_WARMUP
	}
	#endif
#endif
}


#endif /* UPDATE_H_ */
