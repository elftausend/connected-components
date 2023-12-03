extern "C" {
    

    __global__ void setRootLabelIter(ushort4* links, unsigned int* labels, unsigned char* rootCandidates, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        } 
        int outIdx = y * width + x;
        
        unsigned int currentLabel = labels[outIdx];

        ushort4 currentLink = links[outIdx];


        unsigned int farRightIdx = outIdx + (int) currentLink.x;
        unsigned int farDownIdx = (y + currentLink.y) * width + x;
        // unsigned int farLeftIdx = outIdx - currentLink.z;
        // unsigned int farUpIdx = (y - currentLink.w) * width + x;
        
        if (rootCandidates[farRightIdx]) {
            labels[outIdx] = labels[farRightIdx];
        }

        if (rootCandidates[farDownIdx]) {
            labels[outIdx] = labels[farDownIdx];
        }
    }

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
        // label |= connections;

        // right
        for (int i = x + 1; i < width; i++) {
            // break; 
            int4 rightPixel = make_int4(R[y * width + i], G[y * width + i], B[y * width + i], 255);
            if (!(abs(rightPixel.x - currentPixel.x) < threshold && abs(rightPixel.y - currentPixel.y) < threshold && abs(rightPixel.z - currentPixel.z) < threshold)) {
                break;
            }
            unsigned short farRightLink = (unsigned short) (i - x);
            // printf("right link: %d \n", farRightLink);
            currentLink.x = farRightLink;
        }

        // down 
        for (int i = y + 1; i < height; i++) { 
            // break;
            int4 rightPixel = make_int4(R[i * width + x], G[i * width + x], B[i * width + x], 255);
            // printf("%d \n", rightPixel.x);
            if (!(abs(rightPixel.x - currentPixel.x) < threshold && abs(rightPixel.y - currentPixel.y) < threshold && abs(rightPixel.z - currentPixel.z) < threshold)) {
                break;
            }
            unsigned short farDownLabel = (unsigned short) (i - y);
            currentLink.y = farDownLabel;
        }
        // left
        for (int i = x - 1; i >= 0; i--) {
            // break; 
            int4 leftPixel = make_int4(R[y * width + i], G[y * width + i], B[y * width + i], 255);
            if (!(abs(leftPixel.x - currentPixel.x) < threshold && abs(leftPixel.y - currentPixel.y) < threshold && abs(leftPixel.z - currentPixel.z) < threshold)) {
                break;
            }
            unsigned short farLeftLabel = (unsigned short) (x - i);
            // printf("right link: %d \n", farLeftLabel);
            currentLink.z = farLeftLabel;
        }
        // up 
        for (int i = y - 1; i >= 0; i--) { 
            // break;
            int4 rightPixel = make_int4(R[i * width + x], G[i * width + x], B[i * width + x], 255);
            // printf("%d \n", rightPixel.x);
            if (!(abs(rightPixel.x - currentPixel.x) < threshold && abs(rightPixel.y - currentPixel.y) < threshold && abs(rightPixel.z - currentPixel.z) < threshold)) {
                break;
            }
            unsigned short farDownLabel = (unsigned short) (y - i);
            currentLink.w = farDownLabel;
        }

        links[labelIdx] = currentLink;

        // printf("label: %u \n", label);
        labels[labelIdx] = label;

    }
    
    
    // could use bit shifting => store root bit in label
    __global__ void classifyRootCandidates(unsigned int* input, ushort4* links, unsigned char* rootCandidates, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        int outIdx = y * width + x;
        
        unsigned int currentLabel = input[outIdx];
        ushort4 currentLink = links[outIdx];

        // if (currentLink.x == 0 && currentLink.y == 0) {
        //     rootCandidates[outIdx] = 1;
        // }
        unsigned int farRightLabel = input[outIdx + (int) currentLink.x];
        unsigned int farDownLabel = input[(y + currentLink.y) * width + x];

        if (farRightLabel > currentLabel || farDownLabel > currentLabel) {
            rootCandidates[outIdx] = 0;
            return;
        }
        // could use bit shifting => store root bit in label
        rootCandidates[outIdx] = 1;
    }

    __device__ void setRootLinkIfCandidate(unsigned int maybe_root_link_idx, unsigned int currentIdx, unsigned int* rootLinks, unsigned char* rootCandidates) {
        if (rootCandidates[maybe_root_link_idx]) {
            rootLinks[currentIdx] = maybe_root_link_idx;
        }
    }

    
    __global__ void labelComponentsFar(unsigned int* input, unsigned int* out, ushort4* links, int width, int height, int* hasUpdated) {
        // return;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        int outIdx = y * width + x;
        
        unsigned int currentLabel = input[outIdx];

        ushort4 currentLink = links[outIdx];
        unsigned int farRightLabel = input[outIdx + (int) currentLink.x];

        if (farRightLabel > currentLabel) {
            // out[outIdx] = farRightLabel;
            currentLabel = farRightLabel;
            *hasUpdated = 1;
            // atomicOr(hasUpdated, 1);
            // return;
        }
    
        unsigned int farDownLabel = input[(y + currentLink.y) * width + x];

        if (farDownLabel > currentLabel) {
            // out[outIdx] = farDownLabel;
            currentLabel = farDownLabel;
            *hasUpdated = 1;
            // atomicOr(hasUpdated, 1);
            // return;
        }    

        unsigned int farLeftLabel = input[outIdx - currentLink.z];

        if (farLeftLabel > currentLabel) {
            // out[outIdx] = farLeftLabel;
            currentLabel = farLeftLabel;
            *hasUpdated = 1;
            // atomicOr(hasUpdated, 1);
            // return;
        }
     
        
        unsigned int farUpLabel = input[(y - currentLink.w) * width + x];

        if (farUpLabel > currentLabel) {
            // out[outIdx] = farUpLabel;
            currentLabel = farUpLabel;
            *hasUpdated = 1;
            // atomicOr(hasUpdated, 1);
            // return;
        }
        // if (outIdx == 0) {
        //     printf("%d \n", (currentLabel >> 30) & 1);
        // }
        
        int leftLabel = input[outIdx - min(1, currentLink.z)];

        if (leftLabel > currentLabel) {
            // out[outIdx] = farLeftLabel;
            currentLabel = leftLabel;
            *hasUpdated = 1;
        }
  
        int upLabel = input[(y - min(1, currentLink.w)) * width + x];

        if (upLabel > currentLabel) {
            // out[outIdx] = farLeftLabel;
            currentLabel = upLabel;
            *hasUpdated = 1;
        }
 

        // int farUpLabel = input[(y - currentLink.w) * width + x];

        // if (farUpLabel > currentLabel) {
        //     // out[outIdx] = farUpLabel;
        //     currentLabel = farUpLabel;
        //     *hasUpdated = 1;
        //     // atomicOr(hasUpdated, 1);
        //     // return;
        // }
        out[outIdx] = currentLabel;
    }
    
    __global__ void initRootLinks(unsigned int* rootLinks, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        int outIdx = y * width + x;
        rootLinks[outIdx] = outIdx;

    }

}