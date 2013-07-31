/*
 * RenderConfig.h
 *
 *  Created on: May 9, 2013
 *      Author: rr
 */

#ifndef RENDERCONFIG_H_
#define RENDERCONFIG_H_


class Engine;
class RenderConfig;
class Worker;

extern RenderConfig* cfg;

typedef enum engine_t {
	PPM, SPPM, PPMPA, SPPMPA
} engineType;

typedef enum devices_t {
	CPU0 = 100, CPU1=101, GPU0=0, GPU1=1, GPU2=2, GPU3=3
} devicesType;



class RenderConfig {
public:
	RenderConfig() {

		devices = new std::map<devicesType, Worker*>();


		scrRefreshInterval = 1000;
		startTime = 0.0;

		SPPMG_LABEL = (char*) "Many-core Progressive Photon Mapping";

		ndevices=0;

	}
	virtual ~RenderConfig(){

	}

	Engine* GetEngine(){
		return engine;
	}

	engineType GetEngineType(){
			return enginetype;
		}

	std::map<devicesType, Worker*>* devices;

	uint device_configuration;
	uint ndevices;
	engineType enginetype;

	Engine* engine;

	unsigned int photonsFirstIteration;
	unsigned int width;
	unsigned int height;
	unsigned int superSampling;
	float alpha;
	char* SPPMG_LABEL;
	unsigned int hitPointTotal;
	double startTime;
	unsigned int scrRefreshInterval;
	std::string fileName;
	bool rebuildHash;


};

#endif /* RENDERCONFIG_H_ */
