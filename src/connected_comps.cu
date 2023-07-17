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

    __global__ void startlabelComponents(cudaSurfaceObject_t target, int width, int height) {
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

    __global__ void computeLabels(cudaTextureObject_t input, cudaSurfaceObject_t out, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (!(x < width && y < height)) {
            return;
        }

        int threshold = 10;

//        surf2Dwrite(color, target, x * sizeof(uchar4), height -1 - y);
    }
}