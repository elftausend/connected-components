extern "C"{
    
    __global__ void writeToSurface(cudaSurfaceObject_t target, int width, int height, char r, char g, char b) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (x < width && y < height) {
            uchar4 data = make_uchar4(r, g, b, 0xff);
            surf2Dwrite(data, target, x * sizeof(uchar4), y);
        }
    }
    
    __global__ void interleaveRGB(cudaSurfaceObject_t target, int width, int height,
            unsigned char *R, unsigned char *G, unsigned char *B )
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < width && y < height) {       
            unsigned char valR = R[y * width + x]; 
            unsigned char valG = G[y * width + x]; 
            unsigned char valB = B[y * width + x]; 
            uchar4 data = make_uchar4(valR, valG, valB, 0xff);
            surf2Dwrite(data, target, x * sizeof(uchar4), height -1- y);
        }
    }

    __global__ void labelComponents(cudaSurfaceObject_t target, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (!(x < width && y < height)) {
            return;
        }

        float normX = (float) x / (float) width;
        int colorX = (unsigned char) (normX * 255.0f);

        float normY = (float) y / (float) height;
        int colorY = (unsigned char) (normY * 255.0f);

        uchar4 color = make_uchar4(colorX, colorY, 0, 255);

        surf2Dwrite(color, target, x * sizeof(uchar4), height -1 - y);
    }

    __global__ void computeLabels(cudaTextureObject_t input, cudaSurfaceObject_t out, int width, int height, unsigned char *R, unsigned char *G, unsigned char *B) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (!(x < width && y < height)) {
            return;
        }

        __shared__ uchar4 labels[8][32];

        labels[threadIdx.y][threadIdx.x] = tex2D<uchar4>(input, y, x);
        __syncthreads();

        uchar4 label = labels[0][0];

        // shared?
        unsigned char valR = R[y * width + x]; 
        unsigned char valG = G[y * width + x]; 
        unsigned char valB = B[y * width + x]; 

        int threshold = 4;

        if (threadIdx.y+1 < 8) {
            unsigned char valRedRight = R[y * width + x + 1];
            unsigned char valGreenRight = G[y * width + x + 1];
            unsigned char valBlueRight = B[y * width + x + 1];
            uchar4 labelRight = labels[threadIdx.y +1 ][threadIdx.x];
            if (abs(valRedRight - valR) < threshold && abs(valGreenRight - valG) < threshold && abs(valBlueRight - valB) < threshold) {
                surf2Dwrite(labelRight, out, x * sizeof(char4), height -1 - y);
            }   

            if (x == 0 && y == 0) {
                //printf("labelRight: %d %d %d %d\n", labelRight.x, labelRight.y, labelRight.z, labelRight.w);
            }
        }
    
        /*if (threadIdx.y -1 > 0) {
            unsigned char valRedRight = R[y * width + x - 1];
            unsigned char valGreenRight = G[y * width + x - 1];
            unsigned char valBlueRight = B[y * width + x - 1];
            char4 labelRight = labels[threadIdx.y - 1][threadIdx.x];
            if (abs(valRedRight - valR) < threshold && abs(valGreenRight - valG) < threshold && abs(valBlueRight - valB) < threshold) {
                surf2Dwrite(labelRight, out, x * sizeof(char4), height -1 - y);
            }   
        }*/


//        surf2Dwrite(color, target, x * sizeof(uchar4), height -1 - y);
    }
}