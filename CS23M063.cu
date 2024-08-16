#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here

__device__ int c=0;
__constant__ int d_x[1000];
__constant__ int d_y[1000];

__global__ void begin(int round,int *d_health,int *d_score,int T){
    int start=threadIdx.x;
    int end=(start+round)%T;
    
    int x_start=d_x[start];
    int y_start=d_y[start];

    int x_end=d_x[end];
    int y_end=d_y[end];


    int distance=INT_MAX;
    
    if(d_health[start]>0){             
        if(d_health[end]>0){
          
          distance=abs(x_end - x_start) + abs(y_end - y_start);
        }
      
        for(int j=0;j<T;j++){
          int x_j=d_x[j],y_j=d_y[j];

          int temp=abs(x_j - x_start)+ abs(y_j - y_start);
          int condition=((x_end - x_start) * (y_j - y_start)) == ((y_end - y_start) * (x_j - x_start)) && d_health[j] > 0 && temp < distance && j!=start;
          if(condition){
            if (x_end>x_start && x_j>x_start) {
              distance=temp;
              end=j;
            }
            else if (x_end<x_start && x_j<x_start) {
                distance=temp;
                end=j;
            }
            else if (x_end==x_start && y_start<y_end && y_j>y_start) {
                distance=temp;
                end=j;
            }
            else if (x_end==x_start && y_start>y_end && y_j<y_start) {
                distance=temp;
                end=j;
            }
            
          }
          
          
      }

    }
    
    __syncthreads();
    
    
    if(distance!=INT_MAX){
        atomicAdd(&d_score[start], 1);
        atomicSub(&d_health[end], 1);
        
    }
    
    atomicAdd(&c, 1);
}

__global__ void playGame(int *d_health,int *d_score,int T){
    int round=1;
    
    while(1){
        if(round%T!=0){
            begin<<<1,T>>>(round,d_health,d_score,T);
        }
        else{
            atomicAdd(&c, T);
        }
        
        while (atomicCAS(&c, T, 0) < T);
        round+=1;
        int count=0;
        for(int i=0;i<T;i++){
            if(d_health[i]>0){
                count+=1;
            }
            if(count>1){
                break;
            }
        }
        if(count==0 || count==1){
            break;
        }
        
    }

}

__global__ void initializeArray(int *array, int value,int T) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    array[tid] = value;  
}

//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************
    
    int *d_health,*d_score;

    cudaMalloc(&d_health,sizeof(int)*T);
    initializeArray<<<1, T>>>(d_health, H,T);
    cudaDeviceSynchronize();
    cudaMalloc(&d_score,sizeof(int)*T);
    initializeArray<<<1, T>>>(d_score, 0,T);
    cudaDeviceSynchronize();
    cudaMemcpyToSymbol(d_x, xcoord, sizeof(int)*T);
    cudaMemcpyToSymbol(d_y, ycoord, sizeof(int)*T);

    playGame<<<1,1>>>(d_health,d_score,T);
    cudaMemcpy(score,d_score,sizeof(int)*T,cudaMemcpyDeviceToHost);
    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}