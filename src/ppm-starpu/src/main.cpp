#include <cstdlib>
#include <iostream>


#include "utils/config.h"
#include "ppm/engine.h"

#include "unistd.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
  // load configurations
  Config config("Options", argc, argv);

  // load render engine
  ppm::Engine* engine = ppm::Engine::instantiate(config);
  engine->render();
}
