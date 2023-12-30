extern "C" {
    __global__ void labelWithSingleLinks(unsigned int* labels, ushort4* links, unsigned char* R,unsigned char* G,unsigned char* B, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        } 
        int threshold = 20;
        
        uchar4 currentPixel = make_uchar4(R[y * width + x], G[y * width + x], B[y * width + x], 255);
        
        ushort4 currentLink = make_ushort4(0, 0, 0, 0);
        
        // right 
        if (x < width-1) {
            uchar4 rightPixel = make_uchar4(R[y * width + x +1], G[y * width + x +1], B[y * width + x+1], 255);
            // uchar4 rightPixel = img[y * width + x + 1];
             
            if (abs(rightPixel.x - currentPixel.x) < threshold && abs(rightPixel.y - currentPixel.y) < threshold && abs(rightPixel.z - currentPixel.z) < threshold) {    
                currentLink.x = 1;
            }
        }
        // left
        if (x > 0) {
            uchar4 leftPixel = make_uchar4(R[y * width + x -1], G[y * width + x -1], B[y * width + x-1], 255); 
            // uchar4 leftPixel = img[y * width + x - 1];

          if (abs(leftPixel.x - currentPixel.x) < threshold && abs(leftPixel.y - currentPixel.y) < threshold && abs(leftPixel.z - currentPixel.z) < threshold) {    
                currentLink.z = 1;
            }
        }

        if (y < height -1) { 
            // down 
            uchar4 downPixel = make_uchar4(R[(y+1) * width + x ], G[(y +1) * width + x], B[(y+ 1) * width + x], 255);
            // uchar4 downPixel = img[(y + 1) * width + x];

            if (abs(downPixel.x - currentPixel.x) < threshold && abs(downPixel.y - currentPixel.y) < threshold && abs(downPixel.z - currentPixel.z) < threshold) {    
                currentLink.y = 1;
            }
        }
        if (y > 0) { 
            // up
            uchar4 upPixel = make_uchar4(R[(y-1) * width + x ], G[(y -1) * width + x], B[(y- 1) * width + x], 255);
            // uchar4 upPixel = img[(y - 1) * width + x];

            if (abs(upPixel.x - currentPixel.x) < threshold && abs(upPixel.y - currentPixel.y) < threshold && abs(upPixel.z - currentPixel.z) < threshold) {    
                currentLink.w = 1;
            }
        }

        unsigned int labelIdx = y * width + x;

        links[labelIdx] = currentLink;

        
        // links[labelIdx] = currentLink;

        labels[labelIdx] = labelIdx;
    }

    __global__ void globalizeSingleLinkHorizontal(ushort4* links, int width, int height) {
        unsigned int c =threadIdx.x;
        unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

        if (c >= width || r >= height) {
            return;
        }

        __shared__ ushort4 sharedLinks[2048];

        ushort4 currentLink = links[r * width + c];
        ushort4 currentBorderLink;

        sharedLinks[threadIdx.x] = currentLink;

        bool underBorder = threadIdx.x + 1024 < width;
        if (underBorder) {
            currentBorderLink = links[r * width + c + 1024];
            sharedLinks[threadIdx.x + 1024] = currentBorderLink;
        }

        __syncthreads();

        for (int i=0; i<6; i++) {
            if (underBorder && (threadIdx.x + currentBorderLink.x + 1024) < 2048) {
                currentBorderLink.x += sharedLinks[threadIdx.x + 1024 + currentBorderLink.x].x;
            }
            if (underBorder) {
                sharedLinks[threadIdx.x + 1024] = currentBorderLink;
                __syncthreads();
            }


            if (threadIdx.x + currentLink.x < width) {
                // right
                currentLink.x += sharedLinks[threadIdx.x + currentLink.x].x;
            }

            // if ((int)threadIdx.x - (int)currentLink.z >= 0) {
            //     // left
            //     currentLink.z += sharedLinks[threadIdx.x - currentLink.z].z;
            // }

            sharedLinks[threadIdx.x] = currentLink;
            __syncthreads();
        }

        links[r * width + c] = sharedLinks[threadIdx.x];
    }

    __global__ void globalizeSingleLinkVertical(ushort4* links, int width, int height) {
        unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int r = threadIdx.y;

        if (c >= width || r >= height) {
            return;
        }

        __shared__ ushort4 sharedLinks[2048];

        ushort4 currentLink = links[r * width + c];
        ushort4 currentBorderLink;

        bool underBorder = r + 1024 < height;
        if (underBorder) {
            currentBorderLink = links[(r+1024) * width + c];
            sharedLinks[threadIdx.y + 1024] = currentBorderLink;
        }

        sharedLinks[threadIdx.y] = currentLink;
        __syncthreads();

        for (int i=0; i<6; i++) {
            if (underBorder && (threadIdx.y + currentBorderLink.y + 1024) < 2048) {
                currentBorderLink.y += sharedLinks[threadIdx.y + 1024 + currentBorderLink.y].y;
            }

            if (underBorder) {
                sharedLinks[threadIdx.y + 1024] = currentBorderLink;
                __syncthreads();
            }
            
            if (threadIdx.y + currentLink.y < height) {
                // down
                currentLink.y += sharedLinks[threadIdx.y + currentLink.y].y;
                
            }
            // if ((int)threadIdx.y - (int)currentLink.w >= 0) {
            //     // up
            //     currentLink.w += sharedLinks[threadIdx.y - currentLink.w].w;
            // }

            sharedLinks[threadIdx.y] = currentLink;
            __syncthreads();
        }

        links[r * width + c] = sharedLinks[threadIdx.y];
    }

    __global__ void labelWithSharedLinks(unsigned int* labels, ushort4* links, unsigned char* R,unsigned char* G,unsigned char* B, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        } 
        int threshold = 20;
        
        uchar4 currentPixel = make_uchar4(R[y * width + x], G[y * width + x], B[y * width + x], 255);
        
        __shared__ ushort4 sharedLinks[32][33];
        ushort4 currentLink = make_ushort4(0, 0, 0, 0);
        
        // right 
        if (x < width-1) {
            uchar4 rightPixel = make_uchar4(R[y * width + x +1], G[y * width + x +1], B[y * width + x+1], 255);
            // uchar4 rightPixel = img[y * width + x + 1];
             
            if (abs(rightPixel.x - currentPixel.x) < threshold && abs(rightPixel.y - currentPixel.y) < threshold && abs(rightPixel.z - currentPixel.z) < threshold) {    
                currentLink.x = 1;
            }
        }
        // left
        if (x > 0) {
            uchar4 leftPixel = make_uchar4(R[y * width + x -1], G[y * width + x -1], B[y * width + x-1], 255); 
            // uchar4 leftPixel = img[y * width + x - 1];

          if (abs(leftPixel.x - currentPixel.x) < threshold && abs(leftPixel.y - currentPixel.y) < threshold && abs(leftPixel.z - currentPixel.z) < threshold) {    
                currentLink.z = 1;
            }
        }

        if (y < height -1) { 
            // down 
            uchar4 downPixel = make_uchar4(R[(y+1) * width + x ], G[(y +1) * width + x], B[(y+ 1) * width + x], 255);
            // uchar4 downPixel = img[(y + 1) * width + x];

            if (abs(downPixel.x - currentPixel.x) < threshold && abs(downPixel.y - currentPixel.y) < threshold && abs(downPixel.z - currentPixel.z) < threshold) {    
                currentLink.y = 1;
            }
        }
        if (y > 0) { 
            // up
            uchar4 upPixel = make_uchar4(R[(y-1) * width + x ], G[(y -1) * width + x], B[(y- 1) * width + x], 255);
            // uchar4 upPixel = img[(y - 1) * width + x];

            if (abs(upPixel.x - currentPixel.x) < threshold && abs(upPixel.y - currentPixel.y) < threshold && abs(upPixel.z - currentPixel.z) < threshold) {    
                currentLink.w = 1;
            }
        }

        unsigned int labelIdx = y * width + x;

        sharedLinks[threadIdx.y][threadIdx.x] = currentLink;
        __syncthreads();

        for (int i=0; i<5; i++) {
            if (threadIdx.x + currentLink.x < 32) {
                // right
                currentLink.x += sharedLinks[threadIdx.y][threadIdx.x + currentLink.x].x;
            }

            if (threadIdx.y + currentLink.y < 32) {
                // down
                currentLink.y += sharedLinks[threadIdx.y + currentLink.y][threadIdx.x].y;
            }

            if ((int)threadIdx.x - (int)currentLink.z >= 0) {
                // left
                currentLink.z += sharedLinks[threadIdx.y][threadIdx.x - currentLink.z].z;
            }

            if ((int)threadIdx.y - (int)currentLink.w >= 0) {
                // up
                currentLink.w += sharedLinks[threadIdx.y - currentLink.w][threadIdx.x].w;
            }

            sharedLinks[threadIdx.y][threadIdx.x] = currentLink;
            __syncthreads();
        }

        links[labelIdx] = sharedLinks[threadIdx.y][threadIdx.x];

        
        // links[labelIdx] = currentLink;

        labels[labelIdx] = labelIdx;
    }

    __global__ void globalizeLinksVertical(ushort4* links, int active_yd, int active_yu, int width, int height) {
        unsigned int x = x * blockDim.x + threadIdx.x;
        unsigned int yd = active_yd * blockDim.y + threadIdx.y;
        if (x >= width) {
            return;
        } 

        // if (xl < width) {
        //     unsigned short acc_link_z = links[y * width + xl].z;
        //     unsigned short leftMove = acc_link_z;

        //     while (leftMove != 0) {
        //         leftMove = links[y * width + xl - acc_link_z].z;
        //         acc_link_z += leftMove;
        //     }
        //     links[y * width + xl].z = acc_link_z;
        // }

        if (yd < height) {
            unsigned short acc_link_y = links[yd * width + x].y;
            unsigned short downMove = acc_link_y;

            while (downMove != 0) {
                downMove = links[(yd + acc_link_y) * width + x].y;
                acc_link_y += downMove;
            }
            links[yd * width + x].y = acc_link_y;
        }
    }


    __global__ void globalizeLinksHorizontal(ushort4* links, int active_xr, int active_xl, int width, int height) {
        unsigned int xr = active_xr * blockDim.x + threadIdx.x;
        unsigned int xl = active_xl * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y >= height) {
            return;
        } 

        if (xl < width) {
            unsigned short acc_link_z = links[y * width + xl].z;
            unsigned short leftMove = acc_link_z;

            while (leftMove != 0) {
                leftMove = links[y * width + xl - acc_link_z].z;
                acc_link_z += leftMove;
            }
            links[y * width + xl].z = acc_link_z;
        }

        if (xr < width) {
            unsigned short acc_link_x = links[y * width + xr].x;
            unsigned short rightMove = acc_link_x;

            while (rightMove != 0) {
                rightMove = links[y * width + xr + acc_link_x].x;
                acc_link_x += rightMove;
            }
            links[y * width + xr].x = acc_link_x;
        }
    }
    
    __global__ void classifyRootCandidatesShifting(unsigned int* input, ushort4* links, int width, int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        int outIdx = y * width + x;
        
        unsigned int currentLabel = input[outIdx];
        ushort4 currentLink = links[outIdx];

        if (currentLink.x == 0 && currentLink.y == 0) {
            unsigned int rootCandidateLabel = (1 << 31) | currentLabel;
            input[outIdx] = rootCandidateLabel;
            return;
        }

    }
    
    __global__ void labelComponentsFarRootCandidates(unsigned int* input, unsigned int* out, ushort4* links, int width, int height, int* hasUpdated) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        int outIdx = y * width + x;

        int mask = 0b01111111111111111111111111111111;
        
        unsigned int currentLabelRootCandidate = input[outIdx];
        unsigned int is_root_candidate = currentLabelRootCandidate & ~mask;
        // printf("%d", is_root_candidate);
        unsigned int currentLabel = currentLabelRootCandidate & mask; 

        ushort4 currentLink = links[outIdx];


        unsigned int farRightIdx = outIdx + (int) currentLink.x;
        unsigned int farDownIdx = (y + currentLink.y) * width + x;
        unsigned int farLeftIdx = outIdx - currentLink.z;
        unsigned int farUpIdx = (y - currentLink.w) * width + x;
        
        unsigned int farDownLabel = input[farDownIdx] & mask;

        if (farDownLabel > currentLabel) {
            currentLabel = farDownLabel;
            *hasUpdated = 1;
            out[outIdx] =  currentLabel | is_root_candidate;

            // if a larger label was found downwards, it is (probably) larger than the rest
            return;
        }

        if (input[currentLabel] >> 31) {
            // unsigned int rootLabel = input[currentLabel - 1];
            // if (rootLabel > currentLabel) {
            currentLabel = input[currentLabel] & mask;
                // *hasUpdated = 1;

                // out[outIdx] = currentLabel;
                // return;

            // }
        }

        unsigned int farRightLabel = input[farRightIdx] & mask;

        if (farRightLabel > currentLabel) {
            currentLabel = farRightLabel;
            *hasUpdated = 1;
            out[outIdx] =  currentLabel | is_root_candidate;
            return;
        }

        unsigned int farLeftLabel = input[farLeftIdx] & mask;

        if (farLeftLabel > currentLabel) {
            currentLabel = farLeftLabel;
            *hasUpdated = 1;
        }
        
        unsigned int farUpLabel = input[farUpIdx] & mask;

        if (farUpLabel > currentLabel) {
            currentLabel = farUpLabel;
            *hasUpdated = 1;
        }
        
        int leftLabel = input[outIdx - min(1, currentLink.z)] & mask;

        if (leftLabel > currentLabel) {
            currentLabel = leftLabel;
            *hasUpdated = 1;
        }
  
        int upLabel = input[(y - min(1, currentLink.w)) * width + x] & mask;

        if (upLabel > currentLabel) {
            currentLabel = upLabel;
            *hasUpdated = 1;
        }
 
        out[outIdx] = currentLabel | is_root_candidate;
    }

}