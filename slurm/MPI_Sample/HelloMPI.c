#include <stdio.h>
#include <mpi.h>
void main(int argc, char *argv[])
{
    int rank, nprocs, namelen;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &namelen);
    printf("rank %d of %d on %s\n", rank, nprocs, name);
    MPI_Finalize();
}
