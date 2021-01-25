#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    char message[20];
    int myrank, tag=99;
    printf("before init\n");
    MPI_Status status;

    /* Initialize the MPI library */
    MPI_Init(&argc, &argv);
    /* Determine unique id of the calling process of all processes participating
       in this MPI program. This id is usually called MPI rank. */
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {
        strcpy(message, "Hello, there");
        /* Send the message "Hello, there" from the process with rank 0 to the
           process with rank 1. */
        MPI_Send(message, strlen(message)+1, MPI_CHAR, 1, tag, MPI_COMM_WORLD);
    } else {
        /* Receive a message with a maximum length of 20 characters from process
           with rank 0. */
        MPI_Recv(message, 20, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
        printf("received %s\n", message);
    }

    /* Finalize the MPI library to free resources acquired by it. */
    MPI_Finalize();
    return 0;
}