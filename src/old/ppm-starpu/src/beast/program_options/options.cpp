/*
 * options.cpp
 *
 *  Created on: Dec 22, 2012
 *      Author: naps62
 */

#include <beast/program_options/options.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::stringstream;
using std::pair;
using std::make_pair;
using std::ifstream;
using std::vector;

#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>

namespace beast { namespace program_options {

	/**
	 * constructors
	 */
	// default constructor
	options :: options()
	: _options(BEAST_PO_DESC) {
		_init();
	}

	// constructor receiving a custom options description
	options :: options(const char* desc)
	: _options(desc) {
		_init();
	}

	/**
	 * add methods
	 */
	options& options :: flag(
			const char* name,
			bool& var,
			const char* desc) {
		this->_options.add_options() (name, bpo::bool_switch(&var), desc);
		return *this;
	}

	/**
	 *
	 */
	void options::parse(int argc, char **argv) {
		bpo::store(
				bpo::command_line_parser(argc, argv)
				   .options(this->_options)
				   .positional(this->_positional_options)
				   .run()
				, this->vm);

		bpo::notify(this->vm);

		// check for options-file
		if (vm.count(BEAST_PO_OP_FILE_LONG)) {
			vector<string> files = vm[BEAST_PO_OP_FILE_LONG].as<vector<string> >();
			for(vector<string>::reverse_iterator it = files.rbegin(); it != files.rend(); ++it) {
				this->load_options_file(*it);
			}
		}

		// deal with help options
		if (this->count("help")) {
			cout << this->help();
			exit(0);
		}
	}

	/**
	 * Getters
	 */
	bool options::get_flag(const char* name) const {
		return this->get<bool>(name);
	}

	int options::count(const char* name) const {
		return this->vm.count(name);
	}

	bool options::has(const char* name) const {
		return this->count(name) > 0;
	}

	bpo::options_description options::help() const {
		return this->_options;
	}

	void options::_init() {
		this->_options.add_options()
			(BEAST_PO_HELP_OP, BEAST_PO_HELP_DESC)
			(BEAST_PO_OP_FILE, bpo::value<vector<string> >(), BEAST_PO_OP_FILE_DESC);
		this->_positional_options.add(BEAST_PO_OP_FILE_LONG, -1);
	}

//	pair<string, string> options::file_option_parser(const string& s) {
//		if (s[0] == '@')
//			return make_pair(string(BEAST_PO_OP_FILE), s.substr(1));
//		else
//			return pair<string, string>();
//	}

	void options::load_options_file(const string& filename) {
		// Load the file and tokenize it
		ifstream ifs(filename.c_str());
		if (!ifs) {
			cerr << filename << ": " << BEAST_PO_ERROR_OP_FILE << endl;
			exit(1);
		}
		bpo::store(bpo::parse_config_file(ifs, this->_options), this->vm);
		ifs.close();
		bpo::notify(this->vm);
	}

} }
