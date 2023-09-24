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

    __global__ void labelPixels(uchar4* target, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (!(x < width && y < height)) {
            return;
        }

        float normX = (float) x / (float) width;
        unsigned char colorX = (unsigned char) (normX * 255.0f);

        float normY = (float) y / (float) height;
        unsigned char colorY = (unsigned char) (normY * 255.0f);

        // float normZ = (float) (x + y) / (float) (width + height);
        // int colorZ = (unsigned char) (normZ * 255.0f);
        unsigned char colorZ = 0;

        uchar4 color = make_uchar4(colorX, colorY, colorZ, 255);

        target[y * width + x] = color;

        //surf2Dwrite(color, target, x * sizeof(uchar4), height -1 - y);
    }

    __global__ void labelPixelsCombinations(uchar4* target, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (!(x < width && y < height)) {
            return;
        }

        // unsigned char colorZ = (unsigned char) (x + y) / (255*255);

        unsigned char colorZ = x / 255 + (y / 255) * ((width) / 255) + (y / 255);

        uchar4 color = make_uchar4(x % 255, y % 255, colorZ, 255);
        // printf("color: %d %d %d\n", color.x, color.y, color.z);

        target[y * width + x] = color;

        //surf2Dwrite(color, target, x * sizeof(uchar4), height -1 - y);
    }

    __global__ void labelWithConnectionInfo(unsigned int* labels, unsigned char* R,unsigned char* G,unsigned char* B, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        } 
        int threshold = 30;

        unsigned int connections = 0;

        uchar4 currentPixel = make_uchar4(R[y * width + x], G[y * width + x], B[y * width + x], 255);

        // right 
        if (x < width-1) {
            uchar4 rightPixel = make_uchar4(R[y * width + x +1], G[y * width + x +1], B[y * width + x+1], 255);
            // uchar4 rightPixel = img[y * width + x + 1];
            
            if (abs(rightPixel.x - currentPixel.x) < threshold && abs(rightPixel.y - currentPixel.y) < threshold && abs(rightPixel.z - currentPixel.z) < threshold) {    
                connections |= (1u << 31); 
            }
        }
        // left
        if (x > 0) {
            uchar4 leftPixel = make_uchar4(R[y * width + x -1], G[y * width + x -1], B[y * width + x-1], 255); 
            // uchar4 leftPixel = img[y * width + x - 1];

          if (abs(leftPixel.x - currentPixel.x) < threshold && abs(leftPixel.y - currentPixel.y) < threshold && abs(leftPixel.z - currentPixel.z) < threshold) {    
                connections |= (1u << 30); 
            }
        }

        if (y < height -1) { 
            // down 
            uchar4 downPixel = make_uchar4(R[(y+1) * width + x ], G[(y +1) * width + x], B[(y+ 1) * width + x], 255);
            // uchar4 downPixel = img[(y + 1) * width + x];

            if (abs(downPixel.x - currentPixel.x) < threshold && abs(downPixel.y - currentPixel.y) < threshold && abs(downPixel.z - currentPixel.z) < threshold) {    
                connections |= (1u << 29); 
            }
        }
        if (y > 0) { 
            // up
            uchar4 upPixel = make_uchar4(R[(y-1) * width + x ], G[(y -1) * width + x], B[(y- 1) * width + x], 255);
            // uchar4 upPixel = img[(y - 1) * width + x];

            if (abs(upPixel.x - currentPixel.x) < threshold && abs(upPixel.y - currentPixel.y) < threshold && abs(upPixel.z - currentPixel.z) < threshold) {    
                connections |= (1u << 28);
            }
        }        
        unsigned int labelIdx = y * width + x;
        unsigned int label = labelIdx + 1;
        label |= connections;
        // printf("label: %u \n", label);
        labels[labelIdx] = label;

    }

    __global__ void labelComponentsSharedWithConnections(unsigned int* input, unsigned int* out, int width, int height, int threshold, int* hasUpdated, unsigned char offsetY, unsigned char offsetX) {
        unsigned int bloatedBlockIdxX = blockIdx.x * 2 + offsetX;
        unsigned int bloatedBlockIdxY = blockIdx.y * 2 + offsetY;

        unsigned int x = bloatedBlockIdxX * blockDim.x + threadIdx.x;
        unsigned int y = bloatedBlockIdxY * blockDim.y + threadIdx.y;

        // bool threadChanged = false;

        if (x >= width) {
            return;
        }

        if (y >= height) {
            return;
        }

        __shared__ unsigned int labels[34][34];
        // __shared__ bool somethingChanged[1];

        if (threadIdx.y == 0) {
            int upperOverlap = (bloatedBlockIdxY * blockDim.y -1);
            int upperOverlapIdx = upperOverlap * width + x;
            if (upperOverlap >= 0) {

                labels[0][threadIdx.x+1] = input[upperOverlapIdx];

                //maybe not needed, diagonals
                if (threadIdx.x == 0) {
                    labels[0][0] = input[upperOverlapIdx - 1];
                }
                if (threadIdx.x == 31) {
                    labels[0][33] = input[upperOverlapIdx + 1];
                }
            }
        }

        if (threadIdx.y == 31) {
            int lowerOverlap =  (bloatedBlockIdxY * blockDim.y + 32);
            int lowerOverlapIdx = lowerOverlap * width + x;
            if (lowerOverlap < height) {                
                labels[33][threadIdx.x+1] = input[lowerOverlapIdx];

                //maybe not needed, diagonals
                if (threadIdx.x == 0) {
                    labels[33][0] = input[lowerOverlapIdx - 1];
                }
                if (threadIdx.x == 31) {
                    labels[33][33] = input[lowerOverlapIdx + 1];
                }
            }
        }

        if (threadIdx.x == 0) {
            int leftOverlap =  (bloatedBlockIdxX * blockDim.x -1);
            int leftOverlapIdx = y * width + leftOverlap;
            if (leftOverlap >= 0) {
                labels[threadIdx.y+1][0] = input[leftOverlapIdx];
            }
        }

        if (threadIdx.x == 31) {
            int rightOverlap =  (bloatedBlockIdxX * blockDim.x + 32);
            int rightOverlapIdx = y * width + rightOverlap;
            if (rightOverlap < width) {
                labels[threadIdx.y+1][33] = input[rightOverlapIdx];
            }
        }

        labels[threadIdx.y+1][threadIdx.x+1] = input[y * width + x];
        __syncthreads();


        // if (threadIdx.x == 0 && threadIdx.y == 0) {
        //     somethingChanged[0] = 0;
        // }

        // __syncthreads();

        int outIdx = y * width + x;

        unsigned int currentLabel = labels[threadIdx.y+1][threadIdx.x+1];
        // unsigned int currentLabel = labels[threadIdx.x+1][threadIdx.y+1];

        // right        
        {
        unsigned int label = __shfl_down_sync(0xffffffff, currentLabel, 1);
        if (threadIdx.x == 31) {
            label = labels[threadIdx.y+1][threadIdx.x+2];
        }
        // unsigned int label = labels[threadIdx.y+1][threadIdx.x+2];
        // unsigned int currentLabel = __shfl_up_sync(0xffffffff, label, 1);
        // if ((currentLabel & 0b10000000000000000000000000000000) >> 31) {
        if ((currentLabel >> 31) & 1) {
            
            if ((currentLabel & 0x0FFFFFFF) < (label & 0x0FFFFFFF)) {
                labels[threadIdx.y+1][threadIdx.x+1] = (currentLabel & 0xF0000000) | (label & 0x0FFFFFFF);
                // labels[threadIdx.x+1][threadIdx.y+1] = (currentLabel & 0xF0000000) | (label & 0x0FFFFFFF);
                // hasUpdated[0] = 1; 
                // threadChanged = true; 
                atomicOr(hasUpdated, 1);
            }
        }
        }

        // down
        {
        unsigned int label = labels[threadIdx.y+2][threadIdx.x+1];
        // if ((currentLabel & 0b00100000000000000000000000000000) >> 29) {
        if ((currentLabel >> 29) & 1) {
            if ((currentLabel & 0x0FFFFFFF) < (label & 0x0FFFFFFF)) {
                labels[threadIdx.y+1][threadIdx.x+1] = (currentLabel & 0xF0000000) | (label & 0x0FFFFFFF),
                // labels[threadIdx.x+1][threadIdx.y+1] = (currentLabel & 0xF0000000) | (label & 0x0FFFFFFF);
                // threadChanged = true; 
                atomicOr(hasUpdated, 1);
            }
        }
        }
        __syncthreads();

        //left
        {
        unsigned int label = __shfl_up_sync(0xffffffff, currentLabel, 1);
        if (threadIdx.x == 0) {
            label = labels[threadIdx.y+1][threadIdx.x];
        }

        // if ((currentLabel & 0b01000000000000000000000000000000) >> 30) {
        if ((currentLabel >> 30) & 1) {
            if ((currentLabel & 0x0FFFFFFF) < (label & 0x0FFFFFFF)) {
                labels[threadIdx.y+1][threadIdx.x+1] = (currentLabel & 0xF0000000) | (label & 0x0FFFFFFF);
                // labels[threadIdx.x+1][threadIdx.y+1] = (currentLabel & 0xF0000000) | (label & 0x0FFFFFFF);
                // threadChanged = true; 
                atomicOr(hasUpdated, 1);
            }
        }
        }

         // up
        {
        unsigned int label = labels[threadIdx.y][threadIdx.x+1];
        // if ((currentLabel & 0b00010000000000000000000000000000) >> 28) {
        if ((currentLabel >> 28) & 1) {
            if ((currentLabel & 0x0FFFFFFF) < (label & 0x0FFFFFFF)) {
                labels[threadIdx.y+1][threadIdx.x+1] = (currentLabel & 0xF0000000) | (label & 0x0FFFFFFF);
                // labels[threadIdx.x+1][threadIdx.y+1] = (currentLabel & 0xF0000000) | (label & 0x0FFFFFFF);
                atomicOr(hasUpdated, 1);
            }
        }
        }
        __syncthreads();

        // if (threadChanged) {
        //     somethingChanged[0] = true;
        // }

        // __syncthreads();

        // if (threadIdx.x == 0 && threadIdx.y == 0 && somethingChanged[0]) {
        //     *hasUpdated = 1;
        // }

        out[outIdx] = labels[threadIdx.y+1][threadIdx.x+1];
    }
    __global__ void labelPixelsRowed(uchar4* target, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (!(x < width && y < height)) {
            return;
        }

        float normX = (float) x / (float) width;
        int colorX = (unsigned char) (normX * 255.0f);

        uchar4 color = make_uchar4(colorX, 0, 0, 255);

        target[y * width + x] = color;
    }


    __global__ void labelComponentsShared(uchar4* input, uchar4* out, int width, int height, unsigned char* R, unsigned char* G, unsigned char* B, int threshold, int* hasUpdated, unsigned char offsetY, unsigned char offsetX) {
        int bloatedBlockIdxX = blockIdx.x * 2 + offsetX;
        int bloatedBlockIdxY = blockIdx.y * 2 + offsetY;

        int x = bloatedBlockIdxX * blockDim.x + threadIdx.x;
        int y = bloatedBlockIdxY * blockDim.y + threadIdx.y;

        if (x >= width) {
            return;
        }

        if (y >= height) {
            return;
        }

        /*unsigned char activeColor = blockIdx.x % 2 + blockIdx.y % 2 * 2;

        // printf("active color %d\n", activeColor);

        if (color != activeColor) {
            return;
        }*/

        __shared__ uchar4 pixels[34][34];
        __shared__ uchar4 labels[34][34];

        int newY = y;// - blockIdx.y;
        int newX = x;// - blockIdx.x;


        int pixelIdx = y * width + x;

        // int pixelIdx = y * width + x;
        // pixels[threadIdx.y][threadIdx.x] = make_uchar4(R[pixelIdx], G[pixelIdx], B[pixelIdx], 255);
        // labels[threadIdx.y][threadIdx.x] = input[(y-offsetY) * width + (x - offsetX)];
        // labels[threadIdx.y][threadIdx.x] = input[newY * width + newX];

        if (threadIdx.y == 0) {
            int upperOverlap = (bloatedBlockIdxY * blockDim.y -1);
            int upperOverlapIdx = upperOverlap * width + x;
            if (upperOverlap >= 0) {

                labels[0][threadIdx.x+1] = input[upperOverlapIdx];
                pixels[0][threadIdx.x+1] = make_uchar4(R[upperOverlapIdx], G[upperOverlapIdx], B[upperOverlapIdx], 255);

                //maybe not needed, diagonals
                if (threadIdx.x == 0) {
                    labels[0][0] = input[upperOverlapIdx - 1];
                    pixels[0][0] = make_uchar4(R[upperOverlapIdx - 1], G[upperOverlapIdx - 1], B[upperOverlapIdx - 1], 255);
                }
                if (threadIdx.x == 31) {
                    labels[0][33] = input[upperOverlapIdx + 1];
                    pixels[0][33] = make_uchar4(R[upperOverlapIdx + 1], G[upperOverlapIdx + 1], B[upperOverlapIdx + 1], 255);
                }
            }
        }

        if (threadIdx.y == 31) {
            int lowerOverlap =  (bloatedBlockIdxY * blockDim.y + 32);
            int lowerOverlapIdx = lowerOverlap * width + x;
            if (lowerOverlap < height) {                
                labels[33][threadIdx.x+1] = input[lowerOverlapIdx];
                pixels[33][threadIdx.x+1] = make_uchar4(R[lowerOverlapIdx], G[lowerOverlapIdx], B[lowerOverlapIdx], 255);

                //maybe not needed, diagonals
                if (threadIdx.x == 0) {
                    labels[33][0] = input[lowerOverlapIdx - 1];
                    pixels[33][0] = make_uchar4(R[lowerOverlapIdx - 1], G[lowerOverlapIdx - 1], B[lowerOverlapIdx - 1], 255);
                }
                if (threadIdx.x == 31) {
                    labels[33][33] = input[lowerOverlapIdx + 1];
                    pixels[33][33] = make_uchar4(R[lowerOverlapIdx + 1], G[lowerOverlapIdx + 1], B[lowerOverlapIdx + 1], 255);
                }
            }
        }

        if (threadIdx.x == 0) {
            int leftOverlap =  (bloatedBlockIdxX * blockDim.x -1);
            int leftOverlapIdx = y * width + leftOverlap;
            if (leftOverlap >= 0) {
                labels[threadIdx.y+1][0] = input[leftOverlapIdx];
                pixels[threadIdx.y+1][0] = make_uchar4(R[leftOverlapIdx], G[leftOverlapIdx], B[leftOverlapIdx], 255);
            }
        }

        if (threadIdx.x == 31) {
            int rightOverlap =  (bloatedBlockIdxX * blockDim.x + 32);
            int rightOverlapIdx = y * width + rightOverlap;
            if (rightOverlap < width) {
                labels[threadIdx.y+1][33] = input[rightOverlapIdx];
                pixels[threadIdx.y+1][33] = make_uchar4(R[rightOverlapIdx], G[rightOverlapIdx], B[rightOverlapIdx], 255);
            }
        }

        pixels[threadIdx.y+1][threadIdx.x+1] = make_uchar4(R[pixelIdx], G[pixelIdx], B[pixelIdx], 255);
        labels[threadIdx.y+1][threadIdx.x+1] = input[y * width + x];
        // labels[threadIdx.y - blockIdx.y][threadIdx.x - blockIdx.x] = input[y * width + x];        
        __syncthreads();

        // uchar4 currentLabel = labels[threadIdx.y+1][threadIdx.x+1];
        // uchar4 currentPixel = pixels[threadIdx.y+1][threadIdx.x+1];

        // return;
        int outIdx = y * width + x;

        // right        
        {
        uchar4 currentLabel = labels[threadIdx.y+1][32-threadIdx.x];
        uchar4 currentPixel = pixels[threadIdx.y+1][32-threadIdx.x];
        uchar4 pixel = pixels[threadIdx.y+1][33-threadIdx.x];
        uchar4 label = labels[threadIdx.y+1][33-threadIdx.x];
        if (abs(pixel.x - currentPixel.x) < threshold && abs(pixel.y - currentPixel.y) < threshold && abs(pixel.z - currentPixel.z) < threshold) {    
            if ((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) label.x + (int) label.y + (int) label.z) {
                // labels[threadIdx.y][threadIdx.x] = label;
                labels[threadIdx.y+1][32-threadIdx.x] = label;       
                // hasUpdated[0] = 1; 
                atomicOr(hasUpdated, 1);
            }
        }
        }

        __syncthreads();

        // down
        {
        uchar4 currentLabel = labels[32-threadIdx.y][threadIdx.x+1];
        uchar4 currentPixel = pixels[32-threadIdx.y][threadIdx.x+1];

        uchar4 pixel = pixels[33-threadIdx.y][threadIdx.x+1];
        uchar4 label = labels[33-threadIdx.y][threadIdx.x+1];
        if (abs(pixel.x - currentPixel.x) < threshold && abs(pixel.y - currentPixel.y) < threshold && abs(pixel.z - currentPixel.z) < threshold) {    
            if ((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) label.x + (int) label.y + (int) label.z) {
                labels[32-threadIdx.y][threadIdx.x+1] = label;
                atomicOr(hasUpdated, 1);
            }
        }
        }
        __syncthreads();

        //left
        {
        uchar4 currentLabel = labels[threadIdx.y+1][threadIdx.x+1];
        uchar4 currentPixel = pixels[threadIdx.y+1][threadIdx.x+1];
        uchar4 pixel = pixels[threadIdx.y+1][threadIdx.x];
        uchar4 label = labels[threadIdx.y+1][threadIdx.x];

        if (abs(pixel.x - currentPixel.x) < threshold && abs(pixel.y - currentPixel.y) < threshold && abs(pixel.z - currentPixel.z) < threshold) {    
            if ((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) label.x + (int) label.y + (int) label.z) {
                labels[threadIdx.y+1][threadIdx.x+1] = label;
                atomicOr(hasUpdated, 1);
            }
        }
        }
        __syncthreads();

        // up
        {
        uchar4 currentLabel = labels[threadIdx.y+1][threadIdx.x+1];
        uchar4 currentPixel = pixels[threadIdx.y+1][threadIdx.x+1];

        uchar4 pixel = pixels[threadIdx.y][threadIdx.x+1];
        uchar4 label = labels[threadIdx.y][threadIdx.x+1];
        if (abs(pixel.x - currentPixel.x) < threshold && abs(pixel.y - currentPixel.y) < threshold && abs(pixel.z - currentPixel.z) < threshold) {    
            if ((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) label.x + (int) label.y + (int) label.z) {
                labels[threadIdx.y+1][threadIdx.x+1] = label;
                atomicOr(hasUpdated, 1);
            }
        }
        }
        __syncthreads();


        out[outIdx] = labels[threadIdx.y+1][threadIdx.x+1];
    }

    __global__ void labelComponentsMasterLabel(uchar4* input, uchar4* out, int width, int height, unsigned char* R, unsigned char* G, unsigned char* B, int threshold, unsigned char* hasUpdated) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width) {
            return;
        }
        
        if (y >= height) {
            return;
        }

        unsigned char valR = R[y * width + x]; 
        unsigned char valG = G[y * width + x]; 
        unsigned char valB = B[y * width + x]; 

        uchar4 currentLabel = input[y * width + x];
    
        // find most right "master" label

        
        bool hasMaster = false;
        uchar4 masterLabel;
        for (int i = x + 1; i < width; i++) {
            unsigned char valRedRight = R[y * width + i];
            unsigned char valGreenRight = G[y * width + i];
            unsigned char valBlueRight = B[y * width + i];
            uchar4 labelRight = input[y * width + i];
            
            if (!(abs(valRedRight - valR) < threshold && abs(valGreenRight - valG) < threshold && abs(valBlueRight - valB) < threshold)) {            
                break;
            }

            if (((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) labelRight.x + (int) labelRight.y + (int) labelRight.z)) {
                masterLabel = labelRight;
                hasMaster = true;
            }
        }
    
        if (hasMaster) {
            hasMaster = false;
            out[y * width + x] = masterLabel;
            hasUpdated[0] = 1;
            return;
        }

        /*out[y * width + x] = currentLabel;
        return;*/

        for (int i = x - 1; i >= 0; i--) {
            unsigned char valRedLeft = R[y * width + i];
            unsigned char valGreenLeft = G[y * width + i];
            unsigned char valBlueLeft = B[y * width + i];
            uchar4 labelLeft = input[y * width + i];
            
            if (!(abs(valRedLeft - valR) < threshold && abs(valGreenLeft - valG) < threshold && abs(valBlueLeft - valB) < threshold)) {
                break;
            }
            
            if (((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) labelLeft.x + (int) labelLeft.y + (int) labelLeft.z)) {
                masterLabel = labelLeft;
                hasMaster = true;
            }            
        }

        if (hasMaster) {
            hasMaster = false;
            out[y * width + x] = masterLabel;
            hasUpdated[0] = 1;
            return;
        }

        for (int i = y + 1; i < height; i++) {
            unsigned char valRedDown = R[i * width + x];
            unsigned char valGreenDown = G[i * width + x];
            unsigned char valBlueDown = B[i * width + x];
            uchar4 labelDown = input[i * width + x];
            
            if (!(abs(valRedDown - valR) < threshold && abs(valGreenDown - valG) < threshold && abs(valBlueDown - valB) < threshold)) {
                break;
            }

            if ((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) labelDown.x + (int) labelDown.y + (int) labelDown.z) {
                masterLabel = labelDown;
                hasMaster = true;
            }
        }

        if (hasMaster) {
            hasMaster = false;
            out[y * width + x] = masterLabel;
            hasUpdated[0] = 1;
            return;
        }

        for (int i = y - 1; i >= 0; i--) {
            unsigned char valRedUp = R[i * width + x];
            unsigned char valGreenUp = G[i * width + x];
            unsigned char valBlueUp = B[i * width + x];
            uchar4 labelUp = input[i * width + x];
            
            if (!(abs(valRedUp - valR) < threshold && abs(valGreenUp - valG) < threshold && abs(valBlueUp - valB) < threshold)) {
                break;
            }

            if ((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) labelUp.x + (int) labelUp.y + (int) labelUp.z) {
                masterLabel = labelUp;
                hasMaster = true;
            }
        }

        if (hasMaster) {
            hasMaster = false;
            out[y * width + x] = masterLabel;
            hasUpdated[0] = 1;
            return;
        }    

        out[y * width + x] = currentLabel;
    
    }
    
    __global__ void labelComponents(uchar4* input, uchar4* out, int width, int height, unsigned char *R, unsigned char *G, unsigned char *B, int threshold, unsigned char* hasUpdated) {
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

        uchar4 currentLabel = input[y * width + x];

        if (x+1 < width) {
            unsigned char valRedRight = R[y * width + x + 1];
            unsigned char valGreenRight = G[y * width + x + 1];
            unsigned char valBlueRight = B[y * width + x + 1];
            uchar4 labelRight = input[y * width + x + 1];
            
            if (abs(valRedRight - valR) < threshold && abs(valGreenRight - valG) < threshold && abs(valBlueRight - valB) < threshold) {
                
                if ((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) labelRight.x + (int) labelRight.y + (int) labelRight.z) {
                    if (blockIdx.y == 0) {
                        // printf("label right: %d %d %d\n", labelRight.x, labelRight.y, labelRight.z);
                    }
                    // currentLabel = labelRight;

                    hasUpdated[0] = 1;
                    out[y * width + x] = labelRight;
                    //surf2Dwrite(labelRight, out, x * sizeof(char4), height -1 - y);
                    return;
                }
            }   
            //surf2Dwrite(currentLabel, out, x * sizeof(char4), height -1 - y);
        }

        if (x-1 < width) {
            unsigned char valRedRight = R[y * width + x - 1];
            unsigned char valGreenRight = G[y * width + x - 1];
            unsigned char valBlueRight = B[y * width + x - 1];
            uchar4 labelRight = input[y * width + x - 1];
            if (abs(valRedRight - valR) < threshold && abs(valGreenRight - valG) < threshold && abs(valBlueRight - valB) < threshold) {
                
                if ((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) labelRight.x + (int) labelRight.y + (int) labelRight.z) {
                    // currentLabel = labelRight;
                    hasUpdated[0] = 1;
                    out[y * width + x] = labelRight;
                    return;
                }
            }   
        }

        if (y+1 < height) {
            unsigned char valRedBottom = R[(y+1) * width + x];
            unsigned char valGreenBottom = G[(y+1) * width + x];
            unsigned char valBlueBottom = B[(y+1) * width + x];
            uchar4 labelBottom = input[(y+1) * width + x];
            if (abs(valRedBottom - valR) < threshold && abs(valGreenBottom - valG) < threshold && abs(valBlueBottom - valB) < threshold) {
                if ((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) labelBottom.x + (int) labelBottom.y + (int) labelBottom.z) {
                    // currentLabel = labelBottom;
                    hasUpdated[0] = 1;
                    out[y * width + x] = labelBottom;
                    return;
                }
            }
        }
        if (y-1 < height) {
            unsigned char valRedBottom = R[(y-1) * width + x];
            unsigned char valGreenBottom = G[(y-1) * width + x];
            unsigned char valBlueBottom = B[(y-1) * width + x];
            uchar4 labelBottom = input[(y-1) * width + x];
            if (abs(valRedBottom - valR) < threshold && abs(valGreenBottom - valG) < threshold && abs(valBlueBottom - valB) < threshold) {
                if ((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) labelBottom.x + (int) labelBottom.y + (int) labelBottom.z) {
                    // currentLabel = labelBottom;
                    hasUpdated[0] = 1;
                    out[y * width + x] = labelBottom;
                    return;
                }
            }
        }

        out[y * width + x] = currentLabel;
    }

    __global__ void labelComponentsRowed(uchar4* input, uchar4* out, int width, int height, unsigned char *R, unsigned char *G, unsigned char *B) {
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

        int threshold = 30;
        uchar4 currentLabel = input[y * width + x];

        if (x+1 < width) {
            unsigned char valRedRight = R[y * width + x + 1];
            unsigned char valGreenRight = G[y * width + x + 1];
            unsigned char valBlueRight = B[y * width + x + 1];
            uchar4 labelRight = input[y * width + x + 1];
            
            if (abs(valRedRight - valR) < threshold && abs(valGreenRight - valG) < threshold && abs(valBlueRight - valB) < threshold) {
                if ((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) labelRight.x + (int) labelRight.y + (int) labelRight.z) {
                    out[y * width + x] = labelRight;
                    return;
                }
            }   
        }

        if (y+1 < height) {
            unsigned char valRedBottom = R[(y+1) * width + x];
            unsigned char valGreenBottom = G[(y+1) * width + x];
            unsigned char valBlueBottom = B[(y+1) * width + x];
            uchar4 labelBottom = input[(y+1) * width + x];
            if (abs(valRedBottom - valR) < threshold && abs(valGreenBottom - valG) < threshold && abs(valBlueBottom - valB) < threshold) {
                if ((int) currentLabel.x + (int) currentLabel.y + (int) currentLabel.z < (int) labelBottom.x + (int) labelBottom.y + (int) labelBottom.z) {
                    out[y * width + x] = labelBottom;
                    return;
                }
            }
        }

        out[y * width + x] = currentLabel;
    }

    __global__ void copyToSurfaceUnsigned(unsigned int* input, cudaSurfaceObject_t target, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width) {
            return;
        }
        
        if (y >= height) {
            return;
        }
        unsigned int idx = input[y * width + x];
        unsigned int labelIdx = idx & 0x0FFFFFFF;
        // unsigned char blue = labelIdx % 256;
        // unsigned char green = ((labelIdx - blue) / 256) % 256;
        // unsigned char red = ((labelIdx - blue) / (256*256)) - (green / 256);
        unsigned char blue = labelIdx & 255;
        unsigned char green = (labelIdx >> 8) & 255; 
        unsigned char red = (labelIdx >> 16) & 255;
        // unsigned char red = ( (unsigned char) ((double) labelIdx / pow(2.0, 32.0)) * 255.0);
       
        // unsigned char green = ( (unsigned char) ((double) labelIdx / pow(2.0, 16.0)) * 255.0);
        // unsigned char blue = ( (unsigned char) ((double) labelIdx / pow(2.0, 8.0-1.0)) * 255.0);
        surf2Dwrite(make_uchar4(red, green, blue, 255), target, x * sizeof(uchar4), height - 1 - y);
    }
    __global__ void copyToInterleavedBuf(unsigned int* input, uchar4* target, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width) {
            return;
        }
        
        if (y >= height) {
            return;
        }
        unsigned int idx = input[y * width + x];
        unsigned int labelIdx = idx & 0x0FFFFFFF;

        unsigned char blue = labelIdx & 255;
        unsigned char green = (labelIdx >> 8) & 255; 
        unsigned char red = (labelIdx >> 16) & 255;
        // unsigned char red = ( (unsigned char) ((double) labelIdx / pow(2.0, 32.0)) * 255.0);
       
        // unsigned char green = ( (unsigned char) ((double) labelIdx / pow(2.0, 16.0)) * 255.0);
        // unsigned char blue = ( (unsigned char) ((double) labelIdx / pow(2.0, 8.0)) * 255.0);
        // printf("labelid: %d, %d, %d, %d \n", labelIdx, red, green, blue);
        target[y * width + x] = make_uchar4(red, green, blue, 255);
    }
    /*__global__ void readLabelValue(unsigned int* input, uchar4* target, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width) {
            return;
        }
        
        if (y >= height) {
            return;
        }
        unsigned int idx = input[y * width + x];
        unsigned int labelIdx = idx & 0x0FFFFFFF;
        printf("%d", labelIdx);
    }*/
    __global__ void copyToSurface(uchar4* input, cudaSurfaceObject_t target, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width) {
            return;
        }
        
        if (y >= height) {
            return;
        }
        
        uchar4 color = input[y * width + x];
        surf2Dwrite(color, target, x * sizeof(uchar4), height -1 - y);
    }

    __global__ void colorComponentAtPixel(cudaTextureObject_t texture, cudaSurfaceObject_t surface, int posX, int posY, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        __shared__ float4 toSearchLabel;
        if (threadIdx.x == 0) {
            toSearchLabel = tex2D<float4>(texture, posX, height -1 - posY);

        }
        __syncthreads();

        if (x >= width) {
            return;
        }
        
        if (y >= height) {
            return;
        }
        
        // uchar4 label = surf2Dread<uchar4>(surface, x * sizeof(uchar4), height -1 - y);
        float4 label = tex2D<float4>(texture, x, height -1 - y);

        float threshold = 0.019; // 0.019

        // mind that z is not used
        if (abs(toSearchLabel.x - label.x) < threshold && abs(toSearchLabel.y - label.y) < threshold && abs(toSearchLabel.z - label.z) < threshold) {
            uchar4 color = make_uchar4(0, 0, 255, 255);
            surf2Dwrite(color, surface, x * sizeof(uchar4), height -1 - y);
        }
        __syncthreads();

    }

    __global__ void colorComponentAtPixelExact(cudaTextureObject_t texture, cudaSurfaceObject_t surface, int posX, int posY, int width, int height, unsigned char R, unsigned char G, unsigned char B) {
        if (R == 0 && G == 0 && B == 255) {
            return;
        }
        
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width) {
            return;
        }
        
        if (y >= height) {
            return;
        }

        uchar4 toSearchLabel = make_uchar4(R, G, B, 255);
        /*__shared__ uchar4 toSearchLabel;
        if (threadIdx.x == 0) {
            toSearchLabel = surf2Dread<uchar4>(surface, posX * sizeof(uchar4), height -1 - posY);
            // toSearchLabel = tex2D<uchar4>(texture, posX, posY);

        }
        __syncthreads();*/


        // uchar4 label = tex2D<uchar4>(texture, x, y);
        uchar4 label = surf2Dread<uchar4>(surface, x * sizeof(uchar4), height -1 - y);


        if (toSearchLabel.x == label.x && toSearchLabel.y == label.y && toSearchLabel.z == label.z) {
            uchar4 color = make_uchar4(0, 0, 255, 255);
            surf2Dwrite(color, surface, x * sizeof(uchar4), height -1 - y);
        }

    }

    __global__ void readPixelValue(uchar4* labels, int posX, int posY, unsigned char* R, unsigned char* G, unsigned char* B, int width, int height) {

        if (posX >= width) {
            return;
        }

        if (posY >= height) {
            return;
        }

        // uchar4 pixel = tex2D<uchar4>(texture, posX, posY);
        // uchar4 pixel = surf2Dread<uchar4>(surface, posX * sizeof(uchar4), height -1 -posY);
        uchar4 pixel = labels[posY * width + posX];
        R[0] = pixel.x;
        G[0] = pixel.y;
        B[0] = pixel.z;
    }
}