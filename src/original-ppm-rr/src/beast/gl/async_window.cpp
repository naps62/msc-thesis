#include <beast/gl/async_window.hpp>
#include <boost/date_time.hpp>
using namespace std;

#include<GL/freeglut.h>

namespace beast { namespace gl {

	// handle for current opengl thread
	async_window* async_window::current_handle;

	// constructor
	async_window :: async_window(const string _name, const uint _width, const uint _height)
	: has_display(false), name(_name), width(_width), height(_height)
	{
	}

	// destructor
	async_window :: ~async_window() {
	}


	void async_window :: start(const bool wait_for_window) {
		thread = boost::thread(&async_window::run, this);
		if (wait_for_window) {
			boost::unique_lock<boost::mutex> lock(wait_for_window_mut);
			while(!has_display) {
				wait_for_window_cond.wait(lock);
			}
		}
	}


	void async_window :: stop() {
		boost::recursive_mutex::scoped_lock lock(display_mutex);
		if (has_display) {
			glutLeaveMainLoop();
			has_display = false;
		}
	}


	void async_window :: join() {
		thread.join();
	}


	void async_window :: run() {
		async_window::current_handle = this;
		gl_init();
		wait_for_window_cond.notify_all();
		glutMainLoop();
	}



	void async_window :: gl_init() {
		boost::recursive_mutex::scoped_lock lock(display_mutex);

		char* arg0 = new char[15];
		strcpy(arg0, "program_name");
		char* argv[] = { &arg0[0], NULL };
		int argc = 1;

		// glut main inits
		glutInit(&argc, argv);
		glutInitWindowSize(width, height);
		glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
		glutCreateWindow(name.c_str());

		// callbacks
		glutMouseFunc(async_window::_mouse);
		glutKeyboardFunc(async_window::_keyboard);
		glutSpecialFunc(async_window::_special);
		glutDisplayFunc(async_window::_render);
		glutIdleFunc(async_window::_idle);

		// default viewport setup
		glMatrixMode(GL_PROJECTION);
		glViewport(0, 0, width, height);
		glLoadIdentity();
		glOrtho(0.f, width - 1.f,
				0.f, height - 1.f, -1.f, 1.f);
		has_display = true;
	}

	void async_window :: start_display_timer() {
		last_display_timer.start();
	}

	//
	// Callbacks
	//
	void async_window :: render() { }
	void async_window :: keyboard(uchar, int, int) { }
	void async_window :: mouse(int, int, int, int) { }
	void async_window :: special(int, int, int) { }
	void async_window :: idle() { }


	void async_window :: _render() {
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f );
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		async_window::current_handle->render();
		glutSwapBuffers();
		async_window::current_handle->start_display_timer();
	}

	void async_window :: request_update(float max_s) {
		if (last_display_timer.elapsed().s() < max_s) {
			return;
		} else {
			boost::recursive_mutex::scoped_lock lock(display_mutex);
			if (has_display)
				glutPostRedisplay();
		}
	}

	void async_window :: _keyboard(unsigned char key, int mousex, int mousey) {
		switch (key) {
		case 27: // ESC
			async_window::current_handle->stop();
			break;
		}
		// delegate
		async_window::current_handle->keyboard(key, mousex, mousey);
	}

	void async_window :: _mouse(int btn, int state, int x, int y) {
		// delegate
		async_window::current_handle->mouse(btn, state, x, y);
	}

	void async_window :: _special(int key, int mousex, int mousey) {
		// delegate
		async_window::current_handle->special(key, mousex, mousey);
	}

	void async_window :: _idle() {
		// delegate
		async_window::current_handle->idle();
	}

} }
