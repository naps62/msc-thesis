


int main(int argc, char* argv[]) {


	RuntimeScheduler* rs =  new RuntimeScheduler();

	work* w = new work();

	rs->synchronize();
	double start = getTimeMS();
	rs->submit(w);
	rs->synchronize();
	double end = getTimeMS();


	printf("(V) Time GAMA: %.4f\n",(end-start));

	delete rs;

}
