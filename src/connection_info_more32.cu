extern "C" {
    __global__ void labelWithConnectionInfoMore32(unsigned int* labels, ushort4* links, unsigned char* R,unsigned char* G,unsigned char* B, int cycles, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        } 
        int threshold = 20;
        
        unsigned int connections = 0;

        uchar4 currentPixel = make_uchar4(R[y * width + x], G[y * width + x], B[y * width + x], 255);

        
        ushort4 currentLink = make_ushort4(0, 0, 0, 0);

        
        // right 
        if (x < width-1) {
            uchar4 rightPixel = make_uchar4(R[y * width + x +1], G[y * width + x +1], B[y * width + x+1], 255);
            // uchar4 rightPixel = img[y * width + x + 1];
             
            if (abs(rightPixel.x - currentPixel.x) < threshold && abs(rightPixel.y - currentPixel.y) < threshold && abs(rightPixel.z - currentPixel.z) < threshold) {    
                currentLink.x = 1;
                connections |= (1u << 31); 
            }
        }
        // left
        if (x > 0) {
            uchar4 leftPixel = make_uchar4(R[y * width + x -1], G[y * width + x -1], B[y * width + x-1], 255); 
            // uchar4 leftPixel = img[y * width + x - 1];

          if (abs(leftPixel.x - currentPixel.x) < threshold && abs(leftPixel.y - currentPixel.y) < threshold && abs(leftPixel.z - currentPixel.z) < threshold) {    
                currentLink.z = 1;
                connections |= (1u << 30); 
            }
        }

        if (y < height -1) { 
            // down 
            uchar4 downPixel = make_uchar4(R[(y+1) * width + x ], G[(y +1) * width + x], B[(y+ 1) * width + x], 255);
            // uchar4 downPixel = img[(y + 1) * width + x];

            if (abs(downPixel.x - currentPixel.x) < threshold && abs(downPixel.y - currentPixel.y) < threshold && abs(downPixel.z - currentPixel.z) < threshold) {    
                currentLink.y = 1;
                connections |= (1u << 29); 
            }
        }
        if (y > 0) { 
            // up
            uchar4 upPixel = make_uchar4(R[(y-1) * width + x ], G[(y -1) * width + x], B[(y- 1) * width + x], 255);
            // uchar4 upPixel = img[(y - 1) * width + x];

            if (abs(upPixel.x - currentPixel.x) < threshold && abs(upPixel.y - currentPixel.y) < threshold && abs(upPixel.z - currentPixel.z) < threshold) {    
                currentLink.w = 1;
                connections |= (1u << 28);
            }
        }

        unsigned int labelIdx = y * width + x;
        unsigned int label = labelIdx + 1;
        label |= connections;

        // right
        for (int i = x; i < width-x; i++) {
            // break; 
            int4 rightPixel = make_int4(R[y * width + i +1], G[y * width + i +1], B[y * width + i+1], 255);
            if (!(abs(rightPixel.x - currentPixel.x) < threshold && abs(rightPixel.y - currentPixel.y) < threshold && abs(rightPixel.z - currentPixel.z) < threshold)) {
                break;
            }
            unsigned short farRightLink = (unsigned short) (i - x);
            // printf("right link: %d \n", farRightLink);
            currentLink.x = farRightLink;
        }

        // down 
        for (int i = y; i < height-y; i++) { 
            break;
            int4 rightPixel = make_int4(R[(y+ i + 1) * width + x], G[(y + i +1) * width + x], B[(y + i +1) * width + x], 255);
            if (!(abs(rightPixel.x - currentPixel.x) < threshold && abs(rightPixel.y - currentPixel.y) < threshold && abs(rightPixel.z - currentPixel.z) < threshold)) {
                break;
            }
            unsigned short farDownLabel = (unsigned short) i - y;
            currentLink.y = farDownLabel;
        }

        links[labelIdx] = currentLink;

        // printf("label: %u \n", label);
        labels[labelIdx] = label;

    }

    __global__ void labelComponentsFar(unsigned int* input, unsigned int* out, ushort4* links, int width, int height, int* hasUpdated) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        int outIdx = y * width + x;
        
        int currentLabel = input[outIdx];

        ushort4 currentLink = links[outIdx];
        int farRightLabel = input[outIdx + (int) currentLink.x];

        if (farRightLabel > currentLabel) {
            out[outIdx] = farRightLabel;
            *hasUpdated = 1;
            // atomicOr(hasUpdated, 1);
            return;
        }
        
        int farDownLabel = input[(y + currentLink.y) * width + x];

        if (farDownLabel > currentLabel) {
            out[outIdx] = farDownLabel;
            *hasUpdated = 1;
            // atomicOr(hasUpdated, 1);
            return;
        }
        
        int farLeftLabel = input[outIdx - currentLink.z];

        if (farLeftLabel > currentLabel) {
            out[outIdx] = farLeftLabel;
            *hasUpdated = 1;
            // atomicOr(hasUpdated, 1);
            return;
        }
        
        int farUpLabel = input[(y - currentLink.w) * width + x];

        if (farUpLabel > currentLabel) {
            out[outIdx] = farUpLabel;
            *hasUpdated = 1;
            // atomicOr(hasUpdated, 1);
            return;
        }

        out[outIdx] = currentLabel;
    }
 

}