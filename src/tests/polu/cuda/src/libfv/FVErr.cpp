#include <stdlib.h>

#include "FVL/FVErr.h"

namespace FVL {
	FVLog FVErr::err_log(FV_ERRFILE);
	
	void FVErr::error(string &msg, int err_code) {
		output(FV_ERROR, msg);
		exit(err_code);
	}
	
	void FVErr::warn(string &msg) {
		output(FV_WARNING, msg);
	}
	
	void FVErr::output(FV_LogType type, string &msg) {
		stringstream full_msg;
		switch (type) {
			case FV_ERROR:
				full_msg << "Error: ";
				break;
			case FV_WARNING:
				full_msg << "Warning: ";
				break;
			default:
				full_msg << "Other: ";
				break;
		}
	
		full_msg << msg << endl << endl;
		string full_msg_str = full_msg.str();
		err_log  << full_msg_str;
		cerr	 << full_msg_str;
	}
}
