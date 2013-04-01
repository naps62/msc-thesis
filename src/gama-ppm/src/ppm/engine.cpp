#include "ppm/engine.h"
#include "ppm/engines/ppm.h"

// this should be a gama internal
MemorySystem* LowLevelMemAllocator::_memSys = NULL;

namespace ppm {

//
// constructors
//

Engine :: Engine(const Config& _config)
: config(_config), gama(new RuntimeScheduler()), scene(new PtrFreeScene(config)), film(config) {
	ofstream out("gama-ppm.scene.dump");
	out << *scene << endl;
	out.close();

	// load display if necessary
	if (config.use_display) {
		display = new Display(config, film);
		display->start();
		display->wait_for_window();
	}
}

Engine :: ~Engine() {
	// finalize RuntimeSystem
	delete gama;

	// wait for display to close
	if (config.use_display) {
		display->join();
	}
}

//
// public methods
//

// static
Engine* Engine :: get_instance(const Config& config) {
	if (config.engine_name == string("ppm"))
		return new ppm::PPM(config);
	else {
		throw new string("invalid engine name " + config.engine_name);
	}
}

}
