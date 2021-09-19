#include <stdio.h>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char** argv){
  // Init the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print a hello world message
  printf("Hello world from processor %s, rank %d out of %d processors\n",
          processor_name,
          world_rank,
          world_size);

  // sleep for 10 seconds, just enough to see the job executing
  sleep(10);

  // Finalize the MPI environment
  MPI_Finalize();
}
