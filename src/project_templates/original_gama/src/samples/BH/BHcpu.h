/*
 * BHcpu.h
 *
 *  Created on: Aug 1, 2012
 *      Author: ricardo
 */

#ifndef BHCPU_H_
#define BHCPU_H_

#include <float.h>
#include <sys/time.h>
#include <sched.h>

//#define MAXDEPTH 32

//--------------------------------------------
//Data
//--------------------------------------------
static int NBODIES_c, TIMESTEPS_c, NNODES_c, THREADS;
smartPtr<float> gmass, gposx, gposy, gposz, gvelx, gvely, gvelz, gaccx, gaccy, gaccz;
//static volatile float *gmass, *gposx, *gposy, *gposz, *gvelx, *gvely, *gvelz, *gaccx, *gaccy, *gaccz;
static volatile float gradius;
static volatile int /**gchild,*/ *gstart, *gcount/*, *gsort*/;
smartPtr<int> gchild, gsort;
static float dtime_c, dthf_c, epssq_c, itolsq;
static volatile int bottom, gmaxdepth, gstep;

int i;
float runtime;
float rt[6];
struct timeval starttime, endtime;


//--------------------------------------------
//Auxiliary functions
//--------------------------------------------
//inline float min(float a, float b){
//  return (a<b)?a:b;
//}
//
//inline float max(float a, float b){
//  return (a>b)?a:b;
//}
#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0


static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;

__forceinline__ void drndset(int seed) {
	A = 1;
	B = 0;
	randx = (A * seed + B) & MASK;
	A = (MULT * A) & MASK;
	B = (MULT * B + ADD) & MASK;
}

__forceinline__ double drnd() {
	lastrand = randx;
	randx = (A * randx + B) & MASK;
	return (double) lastrand / TWOTO31;
}


//Generates a test input (Identical to the CUDA one)
void genInput(){
  float rsc, vsc, r, x, y, z, sq, scale, v;
  int i;

  drndset(7);
  rsc = (3 * 3.1415926535897932384626433832795) / 16;
  vsc = sqrt(1.0 / rsc);
  for (i = 0; i < NBODIES_c; i++) {
    gmass[i] = 1.0 / NBODIES_c;
    r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1);
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = .5f;//rsc * r / sqrt(sq);
    gposx[i] = x * scale;
    gposy[i] = y * scale;
    gposz[i] = z * scale;

    do {
      x = drnd();
      y = drnd() * 0.1;
    } while (y > x*x * pow(1 - x*x, 3.5));
    v = x * sqrt(2.0 / sqrt(1 + r*r));

    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    gvelx[i] = -1.f*gposx[i];//x * scale;
    gvely[i] = -1.f*gposy[i];//y * scale;
    gvelz[i] = -1.f*gposz[i];//z * scale;
  }
}


void init(){

  gmass = smartPtr<float>(sizeof(float)*(NNODES+1));
  gposx = smartPtr<float>(sizeof(float)*(NNODES+1));
  gposy = smartPtr<float>(sizeof(float)*(NNODES+1));
  gposz = smartPtr<float>(sizeof(float)*(NNODES+1));

  gvelx = smartPtr<float>(sizeof(float)*(NNODES+1));
  gvely = smartPtr<float>(sizeof(float)*(NNODES+1));
  gvelz = smartPtr<float>(sizeof(float)*(NNODES+1));

  gaccx = smartPtr<float>(sizeof(float)*(NNODES+1));
  gaccy = smartPtr<float>(sizeof(float)*(NNODES+1));
  gaccz = smartPtr<float>(sizeof(float)*(NNODES+1));


//  gmass = (float*)malloc(sizeof(float)*(NNODES+1));
//
//  gposx = (float*)malloc(sizeof(float)*(NNODES+1));
//  gposy = (float*)malloc(sizeof(float)*(NNODES+1));
//  gposz = (float*)malloc(sizeof(float)*(NNODES+1));
//
//  gvelx = (float*)malloc(sizeof(float)*(NNODES+1));
//  gvely = (float*)malloc(sizeof(float)*(NNODES+1));
//  gvelz = (float*)malloc(sizeof(float)*(NNODES+1));
//
//  gaccx = (float*)malloc(sizeof(float)*(NNODES+1));
//  gaccy = (float*)malloc(sizeof(float)*(NNODES+1));
//  gaccz = (float*)malloc(sizeof(float)*(NNODES+1));


  gstart = (int*)malloc(sizeof(int)*(NNODES+1));
  gcount = (int*)malloc(sizeof(int)*(NNODES+1));
//  gchild = (int*)malloc(sizeof(int)*(NNODES+1)*8);
//  gsort = (int*)malloc(sizeof(int)*(NNODES+1));
  gchild = smartPtr<int>(sizeof(int)*(NNODES+1)*8);
  gsort = smartPtr<int>(sizeof(int)*(NNODES+1));

  genInput();
}


//--------------------------------------------
//Kernels
//--------------------------------------------

void BoundingBoxKernel()
{
  register int i, k;
  register float minx, miny, minz;
  register float maxx, maxy, maxz;
  float minxt[THREADS], minyt[THREADS], minzt[THREADS];
  float maxxt[THREADS], maxyt[THREADS], maxzt[THREADS];

#pragma omp parallel num_threads(THREADS) default(none) shared(NBODIES_c, THREADS, gposx, gposy, gposz, minxt, minyt, minzt, maxxt, maxyt, maxzt)
  {
    register int i, thid = omp_get_thread_num();
    register long long start, end;
    register float temp;
    register float minx, miny, minz;
    register float maxx, maxy, maxz;

    minx = miny = minz = FLT_MAX;
    maxx = maxy = maxz = FLT_MIN;

    start = (long long)NBODIES_c * thid / THREADS;
    end = (long long)NBODIES_c * (thid + 1) / THREADS;

    for (i = start; i < end; i++) {
      temp = gposx[i];
      minx = min(minx, temp);
      maxx = max(maxx, temp);

      temp = gposy[i];
      miny = min(miny, temp);
      maxy = max(maxy, temp);

      temp = gposz[i];
      minz = min(minz, temp);
      maxz = max(maxz, temp);
    }


    minxt[thid] = minx;
    maxxt[thid] = maxx;
    minyt[thid] = miny;
    maxyt[thid] = maxy;
    minzt[thid] = minz;
    maxzt[thid] = maxz;
  }

  minx = minxt[0];  miny = minyt[0];  minz = minzt[0];
  maxx = maxxt[0];  maxy = maxyt[0];  maxz = maxzt[0];

  for (i = 1; i < THREADS; i++) {
    minx = min(minx, minxt[i]);
    miny = min(miny, minyt[i]);
    minz = min(minz, minzt[i]);

    maxx = max(maxx, maxxt[i]);
    maxy = max(maxy, maxyt[i]);
    maxz = max(maxz, maxzt[i]);
  }

  k = NNODES;
  bottom = k;

  gradius = max(max(maxx - minx, maxy - miny), maxz-minz) * 0.5f;
  gmass[k] = -1.0f;
  gstart[k] = 0;
  gposx[k] = (minx + maxx) * 0.5f;
  gposy[k] = (miny + maxy) * 0.5f;
  gposz[k] = (minz + maxz) * 0.5f;

  k *= 8;
  for (i = 0; i < 8; i++) gchild[k + i] = -1;

  gmaxdepth = 0;
  gstep++;
}

void TreeBuildingKernel()
{
#pragma omp parallel num_threads(THREADS) //default(none) shared(gradius, gposx, gposy, gposz, gchild, gmass, gstart, bottom, gmaxdepth, NNODES, NBODIES_c, THREADS)
  {
    register int thid = omp_get_thread_num();
    register int i = thid, n = 0, j, k, depth, localmaxdepth = 1, lock = 0;
    register int skip = 1, ch, locked, patch, cell, inc;
    register float px, py, pz, r, x, y, z;
    register float radius, rootx, rooty, rootz;

    radius = gradius;
    rootx = gposx[NNODES];
    rooty = gposy[NNODES];
    rootz = gposz[NNODES];

#ifdef __DEBUG
    printf("r = %f  x = %f  y = %f  z = %f\n", radius, rootx, rooty, rootz);
#endif

    while (i < NBODIES_c) {
      if (skip != 0) {
        // new body, so start traversing at root
        skip = 0;
        px = gposx[i];
        py = gposy[i];
        pz = gposz[i];
        n = NNODES;
        j = 0;
        depth = 1;
        r = radius;
        if (rootx < px) j = 1;
        if (rooty < py) j += 2;
        if (rootz < pz) j += 4;
      }

      ch = gchild[n*8+j];
      while (ch >= NBODIES_c) {
        n = ch;
        depth++;
        r *= 0.5f;
        j = 0;
        if (gposx[n] < px) j = 1;
        if (gposy[n] < py) j += 2;
        if (gposz[n] < pz) j += 4;
        ch = gchild[n*8+j];
      }

      if (ch != -2) {  // skip if child pointer is locked and try again later
        locked = n*8+j;
        if (ch == __sync_val_compare_and_swap((int *)&gchild[locked], ch, -2)) {
          if (ch == -1) {
            gchild[locked] = i;
          } else {
            patch = -1;
            do {
              depth++;

              cell = __sync_sub_and_fetch(&bottom, 1);
              //bottom--;
              //cell = bottom;
              patch = max(patch, cell);

              x = (j & 1) * r;
              y = ((j >> 1) & 1) * r;
              z = ((j >> 2) & 1) * r;
              r *= 0.5f;

              gmass[cell] = -1.0f;
              gstart[cell] = -1;
              x = gposx[cell] = gposx[n] - r + x;
              y = gposy[cell] = gposy[n] - r + y;
              z = gposz[cell] = gposz[n] - r + z;
              for (k = 0; k < 8; k++) gchild[cell*8+k] = -1;

              if (patch != cell) {
                gchild[n*8+j] = cell;
              }

              j = 0;
              if (x < gposx[ch]) j = 1;
              if (y < gposy[ch]) j += 2;
              if (z < gposz[ch]) j += 4;
              gchild[cell*8+j] = ch;

              n = cell;
              j = 0;
              if (x < px) j = 1;
              if (y < py) j += 2;
              if (z < pz) j += 4;
              ch = gchild[n*8+j];
            } while(ch >= 0);
            gchild[n*8+j] = i;
            __sync_synchronize();
            gchild[locked] = patch;
          }

          localmaxdepth = max(depth, localmaxdepth);
          i += THREADS;  // move on to next body
          skip = 1;
        }
      }
    }

#pragma omp critical(gmax)
    {
      gmaxdepth = max(gmaxdepth, localmaxdepth);
    }
  }
}

void SummarizationKernel()
{
//#pragma omp parallel num_threads(THREADS) //default(none) shared(bottom, THREADS, NNODES, NBODIES_c, gchild, gmass, gposx, gposy, gposz, gcount)
  {
    register int i, j, k, ch, inc, missing, cnt, lbottom, thid;
    register float m, cm, px, py, pz;
    int childl[8];

    thid = 0;//omp_get_thread_num();
    lbottom = bottom;
    inc = 1;//THREADS;
    k = lbottom + thid;

    // iterate over all cells assigned to thread
    while (k <= NNODES) {
      missing = 0;
      // new cell, so initialize
      cm = 0.0f;
      px = 0.0f;
      py = 0.0f;
      pz = 0.0f;
      cnt = 0;
      j = 0;
      for (i = 0; i < 8; i++) {
        ch = gchild[k*8+i];
        if (ch >= 0) {
          if (i != j) {
            // move children to front (needed later for speed)
            gchild[k*8+i] = -1;
            gchild[k*8+j] = ch;
          }
          childl[missing] = ch;  // cache missing children
          m = gmass[ch];
          missing++;
          if (m >= 0.0f) {
            // child is ready
            missing--;
            if (ch >= NBODIES_c) {  // count bodies (needed later)
              cnt += gcount[ch] - 1;
            }
            // add child's contribution
            cm += m;
            px += gposx[ch] * m;
            py += gposy[ch] * m;
            pz += gposz[ch] * m;
          }
          j++;
        }
      }
      cnt += j;

      while (missing != 0) {
        // poll missing child
        ch = childl[missing - 1];
        m = gmass[ch];
        if (m >= 0.0f) {
          // child is now ready
          missing--;
          if (ch >= NBODIES_c) {
            // count bodies (needed later)
            cnt += gcount[ch] - 1;
          }
          // add child's contribution
          cm += m;
          px += gposx[ch] * m;
          py += gposy[ch] * m;
          pz += gposz[ch] * m;
        }
      }

      // all children are ready, so store computed information
      gcount[k] = cnt;
      m = 1.0f / cm;
      gposx[k] = px * m;
      gposy[k] = py * m;
      gposz[k] = pz * m;
      __sync_synchronize();  // make sure data are visible before setting mass
      gmass[k] = cm;
      k += inc;  // move on to next cell
      //printf("k %d\n",k);
    }
  }
}

void SortKernel()
{
#pragma omp parallel num_threads(THREADS) //default(none) shared(gsort, gcount, gstart, gchild, bottom, THREADS, NNODES, NBODIES_c)
  {
    register int i, k, ch, dec, startl, thid, bottoml;
    thid = omp_get_thread_num();
    dec = THREADS;
    k = NNODES + 1 - dec + thid;
    bottoml = bottom;

    // iterate over all cells assigned to thread
    while (k >= bottoml) {
      startl = gstart[k];
      if (startl >= 0) {
        for (i = 0; i < 8; i++) {
          ch = gchild[k*8+i];
          if (ch >= NBODIES_c) {
            // child is a cell
            gstart[ch] = startl;  // set start ID of child
            startl += gcount[ch];  // add #bodies in subtree
          } else if (ch >= 0) {
            // child is a body
            gsort[startl] = ch;  // record body in 'sorted' array
            startl++;
          }
        }
        k -= dec;  // move on to next cell
      }
    }
  }
}

void ForceCalculationKernel()
{
  register int i, k;
  register int step, maxdepth;
  register float tmp;
  float dq[MAXDEPTH];

  step = gstep;
  maxdepth = gmaxdepth;
  tmp = gradius;
  // precompute values that depend only on tree level
  dq[0] = tmp * tmp * itolsq;
  for (i = 1; i < maxdepth; i++) {
    dq[i] = dq[i - 1] * 0.25f;
    dq[i - 1] += epssq_c;
  }
  dq[i - 1] += epssq_c;

  if (maxdepth <= MAXDEPTH) {
#pragma omp parallel for //num_threads(THREADS) default(none) private(k) shared(NBODIES_c, THREADS, NNODES, dq, step, gmass, gaccx, gaccy, gaccz, gvelx, gvely, gvelz, gposx, gposy, gposz, dthf_c, gstep, gmaxdepth, gradius, gsort, gchild, epssq_c, itolsq) schedule(dynamic, 128)
    for (k = 0; k < NBODIES_c; k++) {
      register int i, n, depth;
      register float px, py, pz, ax, ay, az, dx, dy, dz, tmp;
      int pos[MAXDEPTH], node[MAXDEPTH];

      i = gsort[k];  // get permuted/sorted index
      // cache position info
      px = gposx[i];
      py = gposy[i];
      pz = gposz[i];

      ax = 0.0f;
      ay = 0.0f;
      az = 0.0f;

      // initialize iteration stack, i.e., push root node onto stack
      depth = 0;
      node[0] = NNODES;
      pos[0] = 0;

      while (depth >= 0) {
        // stack is not empty
        while (pos[depth] < 8) {
          // node on top of stack has more children to process
          n = gchild[node[depth]*8+pos[depth]];  // load child pointer
          pos[depth]++;

          if (n >= 0) {
            dx = gposx[n] - px;
            dy = gposy[n] - py;
            dz = gposz[n] - pz;
            tmp = dx*dx + (dy*dy + (dz*dz + epssq_c));  // compute distance squared (plus softening)
            if ((n < NBODIES_c) || (tmp >= dq[depth]) ) {  // check if all threads agree that cell is far enough 	away (or is a body)
              tmp = 1.0f/sqrtf(tmp); // compute distance
              tmp = gmass[n] * tmp * tmp * tmp;
              ax += dx * tmp;
              ay += dy * tmp;
              az += dz * tmp;
            } else {
              // push cell onto stack
              depth++;
              node[depth] = n;
              pos[depth] = 0;
            }
          } else {
            depth = max(0, depth - 1);  // early out because all remaining children are also zero
          }
        }
        depth--;  // done with this level
      } //end while


      if (step > 0) {
        gvelx[i] += (ax - gaccx[i]) * dthf_c;
        gvely[i] += (ay - gaccy[i]) * dthf_c;
        gvelz[i] += (az - gaccz[i]) * dthf_c;
      }

      gaccx[i] = ax;
      gaccy[i] = ay;
      gaccz[i] = az;
    }
  }
}

void IntegrationKernel()
{
  register int i;

#pragma omp parallel for num_threads(THREADS) //default(none) private(i) shared(NBODIES_c, gaccx, gaccy, gaccz, gvelx, gvely, gvelz, gposx, gposy, gposz, dthf_c, dtime_c) schedule(static, 16)
  for (i = 0; i < NBODIES_c; i++) {
    register float dvelx = gaccx[i] * dthf_c;
    register float dvely = gaccy[i] * dthf_c;
    register float dvelz = gaccz[i] * dthf_c;

    register float velhx = gvelx[i] + dvelx;
    register float velhy = gvely[i] + dvely;
    register float velhz = gvelz[i] + dvelz;

    gposx[i] += velhx * dtime_c;
    gposy[i] += velhy * dtime_c;
    gposz[i] += velhz * dtime_c;

    gvelx[i] = velhx + dvelx;
    gvely[i] = velhy + dvely;
    gvelz[i] = velhz + dvelz;
  }
}

//--------------------------------------------
//Main
//--------------------------------------------
int init_bhcpu(){

	  NBODIES_c=NBODIES;

	  if (NBODIES_c > (1 << 30)) {
	    fprintf(stderr, "nbodies is too large: %d\n", NBODIES_c);
	    exit(-1);
	  }
	  NNODES_c = (NBODIES_c * 2) - 1;

	  TIMESTEPS_c=TIMESTEPS;
	  THREADS=4;

	  printf("nbodies = %d, nnodes = %d, TIMESTEPS_c = %d, threads = %d\n", NBODIES_c, NNODES_c, TIMESTEPS_c, THREADS);

	  gstep = -1;
	  gmaxdepth = 1;

	  dtime_c = dtime;  dthf_c = dthf;
	  epssq_c = epssq;
	  itolsq = 1.0f / (0.5 * 0.5);

	  init();

	  for (i = 0; i < 6; i++) {
	    rt[i] = 0.0f;
	  }
}


int bhcpu_main(void)
{


#ifdef __DEBUG
    printf("running kernel 1\n");
#endif
    gettimeofday(&starttime, NULL);
    BoundingBoxKernel();
    gettimeofday(&endtime, NULL);
    runtime = endtime.tv_sec*1000.0 + endtime.tv_usec/1000.0 - starttime.tv_sec*1000.0 - starttime.tv_usec/1000.0;
    rt[0] += runtime;

#ifdef __DEBUG
    printf("running kernel 2\n");
#endif
    gettimeofday(&starttime, NULL);
    TreeBuildingKernel();
    gettimeofday(&endtime, NULL);
    runtime = endtime.tv_sec*1000.0 + endtime.tv_usec/1000.0 - starttime.tv_sec*1000.0 - starttime.tv_usec/1000.0;
    rt[1] += runtime;

#ifdef __DEBUG
    printf("running kernel 3\n");
#endif
    gettimeofday(&starttime, NULL);
    SummarizationKernel();
    gettimeofday(&endtime, NULL);
    runtime = endtime.tv_sec*1000.0 + endtime.tv_usec/1000.0 - starttime.tv_sec*1000.0 - starttime.tv_usec/1000.0;
    rt[2] += runtime;

#ifdef __DEBUG
    printf("running kernel 4\n");
#endif
    gettimeofday(&starttime, NULL);
    SortKernel();
    gettimeofday(&endtime, NULL);
    runtime = endtime.tv_sec*1000.0 + endtime.tv_usec/1000.0 - starttime.tv_sec*1000.0 - starttime.tv_usec/1000.0;
    rt[3] += runtime;

#ifdef __DEBUG
    printf("running kernel 5\n");
#endif
    gettimeofday(&starttime, NULL);
    ForceCalculationKernel();
    gettimeofday(&endtime, NULL);
    runtime = endtime.tv_sec*1000.0 + endtime.tv_usec/1000.0 - starttime.tv_sec*1000.0 - starttime.tv_usec/1000.0;
    rt[4] += runtime;

#ifdef __DEBUG
    printf("running kernel 6\n");
#endif
    gettimeofday(&starttime, NULL);
    IntegrationKernel();
    gettimeofday(&endtime, NULL);
    runtime = endtime.tv_sec*1000.0 + endtime.tv_usec/1000.0 - starttime.tv_sec*1000.0 - starttime.tv_usec/1000.0;
    rt[5] += runtime;


  return 0;
}

#endif /* BHCPU_H_ */
