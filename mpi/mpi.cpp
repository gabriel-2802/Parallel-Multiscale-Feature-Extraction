#include <mpi.h>
#include <memory>
#include "infrastructure/master.h"
#include "infrastructure/worker.h"
#include "infrastructure/entity.h"

using namespace std;

int main(int argc, char** argv) {
	int numtasks, rank;
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	unique_ptr<Entity> entity;

	if (rank == MASTER_RANK) {
		entity = make_unique<Master>(numtasks, rank, "../images/image.png", "../images/output_mpi.png");
	} else {
		entity = make_unique<Worker>(numtasks, rank);
	}

	entity->run();
	
	MPI_Finalize();
	return 0;
}