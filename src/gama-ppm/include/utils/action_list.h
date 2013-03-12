/*
 * edit_action.h
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#ifndef _UTILS_EDIT_ACTION_H_
#define _UTILS_EDIT_ACTION_H_

#include "utils/config.h"

#include <set>
using std::set;

#include <ostream>
using std::ostream;

enum Action {
	ACTION_ERROR,               // invalid action (some error occured)

	ACTION_FILM_EDIT,           // image film resize
	ACTION_CAMERA_EDIT,         // camera parameter editing
	ACTION_GEOMETRY_EDIT,       // dataset related editing
	ACTION_INSTANCE_TRANS_EDIT, // instance transformation related editing
	ACTION_MATERIALS_EDIT,      // material editing
	ACTION_MATERIAL_TYPES_EDIT, // if the type of materials changes
	ACTION_AREA_LIGHTS_EDIT,    // area lights editing
	ACTION_INFINITE_LIGHT_EDIT, // infinite light editing
	ACTION_SUN_LIGHT_EDIT,      // sun light editing
	ACTION_SKY_LIGHT_EDIT,      // sky light editing
	ACTION_TEXTURE_MAPS_EDIT,   // texture maps editing

	ACTION_MAX
};

class ActionList {

public:
	// constructors / destructors
	ActionList();
	~ActionList();

	// reset the list
	void reset();

	// add a single action to the list
	void add_action(const Action action);

	// add all existing actions
	void add_all();

	// removes an action from the list, if it exists
	void remove(const Action action);

	// checks if an action exists in this list
	bool has(const Action action) const;

	// size of the current list
	uint size() const;

	// ostream
	friend ostream& operator<< (ostream& os, const ActionList& list);

private:
	set<Action> actions;  // the list of stored actions

	// convert from int to action
	Action int_to_action(const int i) const;

	// convert from action to string
	string action_to_string(const Action action) const;
};

#endif // _UTILS_EDIT_ACTION_H_
