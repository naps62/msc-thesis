#ifndef _RANDOM_H_
#define _RANDOM_H_

//Cada thread chama rnd(seed). Como seed vai sendo modificado pela função rnd() funciona como uma memória da seed anterior.
//No host, antes de lançar o kernel inicializo um buffer com tantos elementos como o número de threads com a seed inicial para cada thread:
//for( unsigned int i=0; i< size ; ++i ) seeds[i] = mwc();
//
//Se precisar de várias sequências de números aleatórios por tread inicializo vários buffers:
//
//for( unsigned int j=0; j< n_buffers ; ++j ) for( unsigned int i=0; i< size ; ++i ) seeds[j][i] = mwc();


//typedef unsigned int Seed;
//
//
//// Generate random float in [0, 1)
//__host__ __device__ __inline__ unsigned int lcg(unsigned int &prev) {
//   uint LCG_A = 1664525u;
//   uint LCG_C = 1013904223u;
//  prev = (LCG_A * prev + LCG_C);
//  return prev & 0x00FFFFFF;
//}
//
//__host__ __device__ __inline__ unsigned int lcg2(unsigned int &prev) {
//  prev = (prev * 8121 + 28411) % 134456;
//  return prev;
//}
//
//
////#if defined(__CUDACC__)
//__host__ __device__ __inline__ float getFloatRNG(Seed& prev) {
//
//  return ((float) lcg(prev) / (float) 0x01000000);
//
//}
//
////#if defined(__CUDACC__)
//
//__host__ __inline__ float getFloatRNG2(Seed& prev) {
//
//  float scale=RAND_MAX+1.;
//          float base=rand()/scale;
//          float fine=rand()/scale;
//          return base+fine/scale;
//}
//
////#else
//// __inline__ float getFloatRNG(Seed& prev) {
////   float scale=RAND_MAX+1.;
////        float base=rand()/scale;
////        float fine=rand()/scale;
////        return base+fine/scale;}
////#endif
//
//
//
//// Multiply with carry
//__host__ inline uint mwc() {
//  static unsigned long long r[4];
//  static unsigned long long carry;
//  static bool init = false;
//
//  if (!init) {
//    init = true;
//    unsigned int seed = 7654321u, seed0, seed1, seed2, seed3;
//    r[0] = seed0 = lcg2(seed);
//    r[1] = seed1 = lcg2(seed0);
//    r[2] = seed2 = lcg2(seed1);
//    r[3] = seed3 = lcg2(seed2);
//    carry = lcg2(seed3);
//  }
//
//  unsigned long long sum = 2111111111ull * r[3] + 1492ull * r[2] + 1776ull * r[1] + 5115ull
//      * r[0] + 1ull * carry;
//
//  r[3] = r[2];
//  r[2] = r[1];
//  r[1] = r[0];
//  r[0] = static_cast<unsigned int> (sum); // lower half
//  carry = static_cast<unsigned int> (sum >> 32); // upper half
//  return static_cast<unsigned int> (r[0]);
//}


//TauswortheRandomGenerator


#define MASK 0xffffffffUL
#define FLOATMASK 0x00ffffffUL

#define invUI (1.f / (FLOATMASK + 1UL));


typedef struct {
  unsigned int s1, s2, s3;
} Seed;

__host__ __device__ __inline__ unsigned int TAUSWORTHE( unsigned int s,
     unsigned int a,  unsigned int b,  unsigned int c,
     unsigned int d) {
  return ((s & c) << d) ^ (((s << a) ^ s) >> b);
}

__host__ __device__ __inline__ unsigned int validSeed( unsigned int x,
     unsigned int m)  {
  return (x < m) ? (x + m) : x;
}

__host__ __device__ __inline__ unsigned int LCG( unsigned int x)  {
  return x * 69069;
}

__host__ __inline__ Seed mwc( unsigned int seed) {
  Seed s;
  s.s1 = validSeed(LCG(seed), 1);
  s.s2 = validSeed(LCG(s.s1), 7);
  s.s3 = validSeed(LCG(s.s2), 15);

  return s;

}

__host__ __device__ __inline__ unsigned long uintValue(Seed& prev) {
  prev.s1 = TAUSWORTHE(prev.s1, 13, 19, 4294967294UL, 12);
  prev.s2 = TAUSWORTHE(prev.s2, 2, 25, 4294967288UL, 4);
  prev.s3 = TAUSWORTHE(prev.s3, 3, 11, 4294967280UL, 17);

  return (prev.s1 ^ prev.s2 ^ prev.s3);
}

__host__ __device__ __inline__ float getFloatRNG(Seed& prev) {
  return (uintValue(prev) & FLOATMASK) * invUI;
}

#endif
