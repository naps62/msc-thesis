#include "utils/action_list.h"

#include <sstream>
using std::ostringstream;

/*
 * constructors / destructors
 */
ActionList :: ActionList() { }

ActionList :: ~ActionList() { }

// reset the list
void ActionList :: reset() {
	actions.clear();
}

// add a single action to the list
void ActionList :: add_action(const Action action) {
	actions.insert(action);
}

// add all existing actions
void ActionList :: add_all() {
	for(uint i = 0; i < ACTION_MAX; ++i) {
		add_action(int_to_action(i));
	}
}

// removes an action from the list, if it exists
void ActionList :: remove(const Action action) {
	actions.erase(action);
}

// checks if an action exists in this list
bool ActionList :: has(const Action action) const {
	return (actions.find(action) != actions.end());
}

// size of the current list
uint ActionList :: size() const {
	return actions.size();
}

// ostream
ostream& operator<< (ostream& os, const ActionList& list) {
	ostringstream ss;
	ss << "ActionList[ ";
	bool sep = false;
	for (set<Action>::iterator it = list.actions.begin(); it != list.actions.end(); ++it) {
		if (sep) ss << ", ";
		ss << list.action_to_string(*it);
	}
	ss << " ]";
	return os << ss.str();
}

/*
 * private methods
 */

// convert from int to action
Action ActionList :: int_to_action(const int i) const {
	if (i == ACTION_MAX)
		return ACTION_ERROR;
	else
		return static_cast<Action>(i);
}

// convert from action to string
string ActionList :: action_to_string(const Action action) const {
	switch (action) {
	case ACTION_ERROR: return string("ActionError");

	case ACTION_FILM_EDIT:           return string("ActionFilm"); break;
	case ACTION_CAMERA_EDIT:         return string("ActionCamera"); break;
	case ACTION_GEOMETRY_EDIT:       return string("ActionGeometry"); break;
	case ACTION_INSTANCE_TRANS_EDIT: return string("ActionInstanceTrans"); break;
	case ACTION_MATERIALS_EDIT:      return string("ActionMaterials"); break;
	case ACTION_MATERIAL_TYPES_EDIT: return string("ActionMaterialTypes"); break;
	case ACTION_AREA_LIGHTS_EDIT:    return string("ActionAreaLights"); break;
	case ACTION_INFINITE_LIGHT_EDIT: return string("ActionInfiniteLight"); break;
	case ACTION_SUN_LIGHT_EDIT:      return string("ActionSunLight"); break;
	case ACTION_SKY_LIGHT_EDIT:      return string("ActionSkyLight"); break;
	case ACTION_TEXTURE_MAPS_EDIT:   return string("ActionTextureMaps"); break;

	case ACTION_MAX:
	default:
		return string("ActionMax");
		break;
	}
	return string("ActionMax");
}
