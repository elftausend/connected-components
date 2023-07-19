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

    __global__ void labelComponents(uchar4* input, uchar4* out, int width, int height, unsigned char *R, unsigned char *G, unsigned char *B) {
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

        int threshold = 20;
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