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

    __global__ void labelComponents(uchar4* target, int width, int height) {
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

        target[y * width + x] = color;

        //surf2Dwrite(color, target, x * sizeof(uchar4), height -1 - y);
    }

    __global__ void computeLabels(uchar4* input, cudaSurfaceObject_t out, int width, int height, unsigned char *R, unsigned char *G, unsigned char *B) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width) {
            return;
        }
        
        if (y >= height) {
            return;
        }
        

        // shared?
        unsigned char valR = R[y * width + x]; 
        unsigned char valG = G[y * width + x]; 
        unsigned char valB = B[y * width + x]; 

        int threshold = 4;
        uchar4 currentLabel = input[y * width + x];

        if (x+1 < width) {
            unsigned char valRedRight = R[y * width + x + 1];
            unsigned char valGreenRight = G[y * width + x + 1];
            unsigned char valBlueRight = B[y * width + x + 1];
            uchar4 labelRight = input[y * width + x + 1];
            if (abs(valRedRight - valR) < threshold && abs(valGreenRight - valG) < threshold && abs(valBlueRight - valB) < threshold) {
                
                if ((int) currentLabel.x + (int) currentLabel.y < (int) labelRight.x + (int) labelRight.y) {
                    if (blockIdx.y == 0) {
                        printf("label right: %d %d %d\n", labelRight.x, labelRight.y, labelRight.z);
                    }
                    surf2Dwrite(labelRight, out, x * sizeof(char4), height -1 - y);
                    return;
                }
            }   
            surf2Dwrite(currentLabel, out, x * sizeof(char4), height -1 - y);
        }
    }
}