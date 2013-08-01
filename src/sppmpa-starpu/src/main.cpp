#include <cstdlib>
#include <iostream>


#include "utils/config.h"
#include "ppm/starpu_engine.h"

#include "unistd.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
  // load configurations
  Config config("Options", argc, argv);

  ppm::Engine* engine;
  // load render engine
  switch(config.engine) {
    //case 1: engine = new ppm::CPUEngine(config);
    //case 2: engine = new ppm::CUDAEngine(config);
    case 0:
    default: engine = new ppm::StarpuEngine(config);
  }
  //  ppm::Engine engine(config, 1);
  engine->render();
  engine->output();

  delete engine;
}
