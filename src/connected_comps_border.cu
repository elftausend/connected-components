
extern "C" {

    __global__ void labelWithSharedLinksInterleaved(unsigned int* labels, ushort4* links, uchar4* pixels, int width, int height) {
        unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
        if (c >= width || r >= height) {
            return;
        }
        int threshold = 20;

        unsigned int idx = r * width + c;

        uchar4 currentPixel = pixels[idx];
        // if (currentPixel.x == 255 && currentPixel.y == 255 && currentPixel.z == 255) {
        //     return;
        // }

        __shared__ ushort4 sharedLinks[32][33];
        ushort4 currentLink = make_ushort4(0, 0, 0, 0);

        unsigned int rightMove = c < width - 1 ? 1 : 0;
        unsigned int downMove = r < height - 1 ? 1 : 0;
        unsigned int leftMove = c > 0 ? 1 : 0;
        unsigned int upMove = r > 0 ? 1 : 0;

        uchar4 rightPixel = pixels[idx + rightMove];
        uchar4 downPixel = pixels[idx + downMove * width];
        uchar4 leftPixel = pixels[idx - leftMove];
        uchar4 upPixel = pixels[idx - upMove * width];

        int rightPixelDifferenceSum = abs(rightPixel.x - currentPixel.x) + abs(rightPixel.y - currentPixel.y) + abs(rightPixel.z - currentPixel.z);  
        if (rightPixelDifferenceSum < threshold && rightMove) {    
            currentLink.x = 1;
        }

        int downPixelDifferenceSum = abs(downPixel.x - currentPixel.x) + abs(downPixel.y - currentPixel.y) + abs(downPixel.z - currentPixel.z);
        if (downPixelDifferenceSum < threshold && downMove) {    
            currentLink.y = 1;
        }

        int leftPixelDifferenceSum = abs(leftPixel.x - currentPixel.x) + abs(leftPixel.y - currentPixel.y) + abs(leftPixel.z - currentPixel.z);
        if (leftPixelDifferenceSum < threshold && leftMove) {    
            currentLink.z = 1;
        }

        int upPixelDifferenceSum = abs(upPixel.x - currentPixel.x) + abs(upPixel.y - currentPixel.y) + abs(upPixel.z - currentPixel.z);
        if (upPixelDifferenceSum < threshold && upMove) {    
            currentLink.w = 1;
        }

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

        links[idx] = sharedLinks[threadIdx.y][threadIdx.x];

        
        // links[labelIdx] = currentLink;

        labels[idx] = idx;
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

    struct PossibleDirections {
        char right;
        char down;
        char left;
        char up;
    };

    __device__ PossibleDirections possibleDirections(char lastDir, ushort4 borderLinks, unsigned int borderLinkIdx, unsigned int* hasVisited, int width, int height) {
            char rightIsPossible = borderLinks.x >= 1 && (borderLinks.y == 0 || borderLinks.w == 0) && lastDir != 2;
            char downIsPossible = borderLinks.y >= 1 && (borderLinks.x == 0 || borderLinks.z == 0) && lastDir != 3;
            char leftIsPossible = borderLinks.z >= 1 && (borderLinks.y == 0 || borderLinks.w == 0) && lastDir != 0;
            char upIsPossible = borderLinks.w >= 1 && (borderLinks.x == 0 || borderLinks.z == 0) && lastDir != 1;
            char isMovePossible = rightIsPossible || downIsPossible || leftIsPossible || upIsPossible;

            // unsigned int nextBorderLinkIdxRight = borderLinkIdx + 1;
            // unsigned int nextBorderLinkIdxDown = borderLinkIdx + width;
            // unsigned int nextBorderLinkIdxLeft = borderLinkIdx - 1;
            // unsigned int nextBorderLinkIdxUp = borderLinkIdx - width;

            return PossibleDirections{rightIsPossible, downIsPossible, leftIsPossible, upIsPossible};

    }

    // use mask and shifting for hasVisited
    __global__ void createBorderPath(unsigned int* labels, ushort4* links, unsigned char* hasVisited, int width, int height) {
        unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

        if (c >= width || r >= height) {
            return;
        }

        int outIdx = r * width + c;
        unsigned int currentLabel = labels[outIdx];

        if (!(currentLabel >> 31)) {
            return;
        }
        unsigned int mask = 0b01111111111111111111111111111111;
        unsigned int rootCandidateLabel = currentLabel & mask;
        unsigned int firstBorderLinkIdx = outIdx;
        unsigned int borderLinkIdx = firstBorderLinkIdx;

        ushort4 borderLinks = links[borderLinkIdx];

        // labels[borderLinkIdx] = 233550;

        char done = false; 
        char lastDir = 1;
        char state = 0;

        int iters = 0;
        while (!done) {

            char rightIsPossible;
            char downIsPossible;
            char leftIsPossible;
            char upIsPossible;

            if (state == 0) {
                rightIsPossible = borderLinks.x >= 1 && (borderLinks.y == 0 || borderLinks.w == 0) && lastDir != 2;
                downIsPossible = borderLinks.y >= 1 && (borderLinks.x == 0 || borderLinks.z == 0) && lastDir != 3;
                leftIsPossible = borderLinks.z >= 1 && (borderLinks.y == 0 || borderLinks.w == 0) && lastDir != 0;
                upIsPossible = borderLinks.w >= 1 && (borderLinks.x == 0 || borderLinks.z == 0) && lastDir != 1;

                char isMovePossible = rightIsPossible || downIsPossible || leftIsPossible || upIsPossible;
                if (!isMovePossible) {
                    ushort4 nextLinksLeft = links[borderLinkIdx - 1];
                    leftIsPossible = nextLinksLeft.x >= 1 && (nextLinksLeft.y == 0 || nextLinksLeft.w == 0) && (lastDir == 3 || lastDir == 1);

                    ushort4 nextLinksRight = links[borderLinkIdx + 1];
                    rightIsPossible = nextLinksRight.z >= 1 && (nextLinksRight.y == 0 || nextLinksRight.w == 0) && (lastDir == 3 || lastDir == 1);
                    
                    ushort4 nextLinksDown = links[borderLinkIdx + width];
                    downIsPossible = nextLinksDown.w >= 1 && (nextLinksDown.x == 0 || nextLinksDown.z == 0) && (lastDir == 0 || lastDir == 2);
                    
                    ushort4 nextLinksUp = links[borderLinkIdx - width];
                    upIsPossible = nextLinksUp.y >= 1 && (nextLinksUp.x == 0 || nextLinksUp.z == 0) && (lastDir == 0 || lastDir == 2);

                    isMovePossible = rightIsPossible || downIsPossible || leftIsPossible || upIsPossible;
                    if (isMovePossible) {
                        // printf("%d %d %d %d\n", rightIsPossible, downIsPossible, leftIsPossible, upIsPossible);
                    }
                }
                // rightIsPossible = borderLinks.x >= 1;
                // downIsPossible = borderLinks.y >= 1;
                // leftIsPossible = borderLinks.z >= 1;
                // upIsPossible = borderLinks.w >= 1;
            }

            char isMovePossible = rightIsPossible || downIsPossible || leftIsPossible || upIsPossible;
            unsigned int nextBorderLinkIdxRight = borderLinkIdx + 1;
            unsigned int nextBorderLinkIdxDown = borderLinkIdx + width;
            unsigned int nextBorderLinkIdxLeft = borderLinkIdx - 1;
            unsigned int nextBorderLinkIdxUp = borderLinkIdx - width;

            if (rightIsPossible && (!hasVisited[nextBorderLinkIdxRight] || state == 1)) {
                lastDir = 0;
                borderLinkIdx = borderLinkIdx + 1;
                borderLinks = links[borderLinkIdx];
            } else if (downIsPossible && (!hasVisited[nextBorderLinkIdxDown] || state == 1)) {
                lastDir = 1;
                borderLinkIdx = borderLinkIdx + width;
                borderLinks = links[borderLinkIdx];
            } else if (leftIsPossible && (!hasVisited[nextBorderLinkIdxLeft] || state == 1)) {
                lastDir = 2;
                borderLinkIdx = borderLinkIdx - 1;
                borderLinks = links[borderLinkIdx];
            } else if ((upIsPossible && !hasVisited[nextBorderLinkIdxUp] || state == 1)) {
                lastDir = 3;
                borderLinkIdx = borderLinkIdx - width;
                borderLinks = links[borderLinkIdx];
            } else {
                // state = 1;
                // printf("jun");
                done = true;
            }
            if (borderLinkIdx >= firstBorderLinkIdx) {
                hasVisited[firstBorderLinkIdx] = 0;
                hasVisited[borderLinkIdx] = 0;
                break;
            }

            // printf("doin iter %d \n", iters);
            // labels[borderLinkIdx] = 
            atomicMax(&labels[borderLinkIdx], rootCandidateLabel);

            // if (iters >= 1000) {
            //     done = true;
            // }

            // unsigned int hasVisited = 0x40000000; 


            // if (rightIsPossible) {
            //     lastDir = 0;
            //     borderLinkIdx = borderLinkIdx + 1;
            //     borderLinks = links[borderLinkIdx];
            // } else {
            //     // junction, dead end, connection
            //     lastDir = 255;
            // }

            // if (!isMovePossible) {
            //     // junction, dead end
            //     lastDir = 255;
            // }

            // if (rightIsPossible) {
            //     lastDir = 0;
            //     borderLinkIdx = borderLinkIdx + 1;
            //     borderLinks = links[borderLinkIdx];
            // } else if (downIsPossible) {
            //     lastDir = 1;
            //     borderLinkIdx = borderLinkIdx + width;
            //     borderLinks = links[borderLinkIdx];
            // } else if (leftIsPossible) {
            //     lastDir = 2;
            //     borderLinkIdx = borderLinkIdx - 1;
            //     borderLinks = links[borderLinkIdx];
            // } else if (upIsPossible) {
            //     lastDir = 3;
            //     borderLinkIdx = borderLinkIdx - width;
            //     borderLinks = links[borderLinkIdx];
            // } else {
            //     printf("jun");
            //     done = true;
            //     // lastDir = 255;
            // }     

            // if (iters > 5000) {
            //     done = true;
            // }
            // labels[borderLinkIdx] = 233550;
            
            iters += 1;

            hasVisited[borderLinkIdx] = 1;
        }

        // labels[firstBorderLinkIdx] = 233550;
    }

    __global__ void fetchFromBorder(unsigned int* labels, ushort4* links, int width, int height) {
        unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

        if (c >= width || r >= height) {
            return;
        }

        int outIdx = r * width + c;
        unsigned int currentLabel = labels[outIdx];

        ushort4 currentLinks = links[outIdx];
        unsigned int mask = 0b01111111111111111111111111111111;
        
        unsigned int farDownLabel = labels[(r + currentLinks.y) * width + c] & mask;
        unsigned int farRightLabel = labels[(r) * width + c + currentLinks.x] & mask;
        unsigned int farLeftLabel = labels[(r) * width + c - currentLinks.z] & mask;
        unsigned int farUpLabel = labels[(r - currentLinks.w) * width + c] & mask;

        labels[outIdx] = max(max(max(farRightLabel, farUpLabel), farDownLabel), farLeftLabel);
        // labels[outIdx] = max(max(max(maybeRootViaDown, farUpLabel), farDownLabel), farLeftLabel);
    }
    
}