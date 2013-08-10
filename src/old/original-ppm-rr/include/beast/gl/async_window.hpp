/*
 * async_window.h
 *
 *  Created on: Feb 10, 2013
 *      Author: naps62
 */

#ifndef _BEAST_GL_ASYNC_WINDOW_HPP_
#define _BEAST_GL_ASYNC_WINDOW_HPP_

#include <beast/common.hpp>
#include <beast/time/timer.hpp>

#include <boost/thread.hpp>
#include <string>
using std::string;

namespace beast { namespace gl {

	/*
	 * OpenGL window manager running asynchronously on a dedicated thread
	 * assumes there is only one OpenGL instance running
	 */
	class async_window {

		// opengl calls need to be static. A handle to the current async_gl instance is required
		// currently assumes only one instance is used at a time
		static async_window* current_handle;


	public:
		// constructor
		async_window(const string _name, const uint _width, const uint _height);

		// destructor
		virtual ~async_window();

		void start(const bool wait_for_window = false);
		void stop();
		void join();
		void run();

		/*
		 * opengl public (overridable) callbacks
		 * override these methods to add functionality to each event
		 */
		virtual void keyboard(uchar, int mousex, int mousey);
		virtual void mouse(int btn, int state, int x, int y);
		virtual void special(int key, int mousex, int mousey);
		virtual void idle();
		virtual void render();

		virtual void request_update(float max_ns);

	protected:
		boost::thread thread;
		boost::recursive_mutex display_mutex;
		boost::mutex wait_for_window_mut;
		boost::condition_variable wait_for_window_cond;
		beast::time::timer last_display_timer;
		bool has_display;
		const string name;
		const int width;
		const int height;

		void gl_init();
		void start_display_timer();

		/*
		 * opengl private callbacks
		 */
		static void _keyboard(unsigned char, int, int);
		static void _mouse(int, int, int, int);
		static void _special(int, int, int);
		static void _idle();
		static void _render();
	};

} }


#endif // _BEAST_GL_ASYNC_WINDOW_HPP_
