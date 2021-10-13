#include <stdio.h>
#include <omp.h>
void main() 
{
#pragma omp parallel
{
    printf("tid=%d\n", omp_get_thread_num());
}
}
