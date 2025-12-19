#include <mpi.h>
#include <memory>
#include <chrono>

#include "infrastructure/master.h"
#include "infrastructure/worker.h"
#include "infrastructure/entity.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {

	int numtasks, rank;
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	unique_ptr<Entity> entity;

	if (rank == MASTER_RANK) {
		auto start = high_resolution_clock::now();

		entity = make_unique<Master>(numtasks, rank, "../images/upscaled_image.png", "../images/output_mpi.png");
		entity->run();

		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(stop - start);
		cout << "Processing time: " << duration.count() << " ms" << endl;

	} else {
		entity = make_unique<Crew>(numtasks, rank);
		entity->run();
	}

	auto stop = high_resolution_clock::now();
	MPI_Finalize();

	return 0;
}