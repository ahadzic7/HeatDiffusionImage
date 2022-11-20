#include <mpi.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <tuple>
#include <iostream>
#include <cmath>

using namespace std;
using namespace std::chrono;

constexpr double DEFAULT_TEMP = 0;
constexpr double EPS_DIFF = 1e-5;
constexpr int ROOT = 0;

int WORLD_SIZE = -1;
int RANK = -2;

int ROOT_HEIGHT = -3;
int PROCESS_HEIGHT = -4;

int HEIGHT = -5;
int WIDTH = -6;

int ITERATION = 0;

vector<int> chunkSizes;
vector<int> displ;

// Spot with permanent temperature on coordinates [x,y].
struct Spot {
    int mX, mY;
    float mTemperature;
    bool operator==(const Spot &b) const { return (mX == b.mX) && (mY == b.mY); }
};

tuple<int, int, vector<Spot>> readInstance(const string &instanceFileName) {
    int width, height;
    vector<Spot> spots;
    string line;

    ifstream file(instanceFileName);
    if (file.is_open()) {
        int lineId = 0;
        while (std::getline(file, line)) {
            stringstream ss(line);
            if (lineId == 0) {
                ss >> width;
            } else if (lineId == 1) {
                ss >> height;
            } else {
                int i, j, temperature;
                ss >> i >> j >> temperature;
                spots.push_back({i, j, (float)temperature});
            }
            lineId++;
        }
        file.close();
    } else {
        throw runtime_error("It is not possible to open instance file!\n");
    }
    return make_tuple(width, height, spots);
}

void writeOutput(const int myRank, const int width, const int height, const string &outputFileName, const vector<float> &temperatures) {
    // Draw the output image in Netpbm format.
    ofstream file(outputFileName);
    if (file.is_open()) {
        if (myRank == 0) {
            file << "P2\n" << width << "\n" << height << "\n" << 255 << "\n";
            for (auto temperature: temperatures) {
                file << (int)max(min(temperature, 255.0f), 0.0f) << " ";
            }
        }
    }
}

void printHelpPage(char *program) {
    cout << "Simulates a simple heat diffusion." << endl;
    cout << endl << "Usage:" << endl;
    cout << "\t" << program << " INPUT_PATH OUTPUT_PATH" << endl << endl;
}

//-----------------------------------My functions------------------

//initialization

void rootInit(vector<vector<float>> &map, vector<vector<float>> &newMap, vector<vector<bool>> &heatSource, const vector<Spot> &spots) {
    map.resize(HEIGHT, vector<float>(WIDTH, DEFAULT_TEMP));
    newMap.resize(HEIGHT, vector<float>(WIDTH, DEFAULT_TEMP));
    heatSource.resize(HEIGHT, vector<bool>(WIDTH, false));

    for (const auto &spot : spots) {
        map[spot.mX][spot.mY] = spot.mTemperature;
        newMap[spot.mX][spot.mY] = spot.mTemperature;
        heatSource[spot.mX][spot.mY] = true;
    }

    //bcast the problem dimensions to subprocesses
    int dimensions[3] = {PROCESS_HEIGHT, WIDTH, ROOT_HEIGHT};
    MPI_Bcast(dimensions, 3, MPI_INT, ROOT, MPI_COMM_WORLD);

    chunkSizes.resize(WORLD_SIZE, PROCESS_HEIGHT * WIDTH);
    chunkSizes[0] = ROOT_HEIGHT * WIDTH;

    displ.resize(WORLD_SIZE, 0);
    int sum = 0;
    for(int i = 0; i < WORLD_SIZE; i++) {
        displ[i] = sum;
        sum += chunkSizes[i];
    }

    int size(WIDTH * HEIGHT);
    vector<float> mapBcast(size, 0);
    vector<int> hsBcast(size, 0);

    int k = 0;
    for(int i = 0; i < HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
            mapBcast[k] = map[i][j];
            hsBcast[k] = heatSource[i][j];
            k++;
        }
    }
    vector<float> rcv(chunkSizes[RANK], 0);

    MPI_Scatterv(mapBcast.data(), chunkSizes.data(), displ.data(), MPI_FLOAT, rcv.data(), chunkSizes[RANK], MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Scatterv(hsBcast.data(), chunkSizes.data(), displ.data(), MPI_INT, rcv.data(), chunkSizes[RANK], MPI_INT, ROOT, MPI_COMM_WORLD);
}

void subProcesInit(vector<vector<float>> &map, vector<vector<float>> &newMap, vector<vector<bool>> &heatSource) {
    vector<int> ibuffer(3);
    MPI_Bcast(ibuffer.data(),ibuffer.size(), MPI_INT,ROOT,MPI_COMM_WORLD);

    PROCESS_HEIGHT = ibuffer[0];
    WIDTH = ibuffer[1];
    ROOT_HEIGHT = ibuffer[2];

    chunkSizes.resize(WORLD_SIZE, PROCESS_HEIGHT * WIDTH);
    chunkSizes[0] = ROOT_HEIGHT * WIDTH;

    displ.resize(WORLD_SIZE, 0);
    int sum = 0;
    for(int i = 0; i < WORLD_SIZE; i++) {
        displ[i] = sum;
        sum += chunkSizes[i];
    }

    if(RANK == WORLD_SIZE - 1)
    {
        map.resize(PROCESS_HEIGHT + 1, vector<float>(WIDTH, DEFAULT_TEMP));
        newMap.resize(PROCESS_HEIGHT + 1, vector<float>(WIDTH, DEFAULT_TEMP));
        heatSource.resize(PROCESS_HEIGHT + 1, vector<bool>(WIDTH, false));
    }
    else {
        map.resize(PROCESS_HEIGHT + 2, vector<float>(WIDTH, DEFAULT_TEMP));
        newMap.resize(PROCESS_HEIGHT + 2, vector<float>(WIDTH, DEFAULT_TEMP));
        heatSource.resize(PROCESS_HEIGHT + 2, vector<bool>(WIDTH, false));
    }

    vector<float>matrixBuff(chunkSizes[RANK], 0);
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_FLOAT, matrixBuff.data(), chunkSizes[RANK], MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    vector<int>hsBuff(chunkSizes[RANK], 0);
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, hsBuff.data(), chunkSizes[RANK], MPI_INT, ROOT, MPI_COMM_WORLD);


    int k = 0;
    for(int i = 1; i < PROCESS_HEIGHT + 1; i++) {
        for(int j = 0; j < WIDTH; j++) {
            map[i][j] = matrixBuff[k];
            heatSource[i][j] = hsBuff[k];
            k++;
        }
    }

}

//communications
MPI_Status status;
MPI_Request request;

void sendRow(int move, const vector<float> &row) {
    MPI_Send(row.data(),WIDTH, MPI_FLOAT,RANK + move,0,MPI_COMM_WORLD);
    //MPI_Isend(row.data(),WIDTH, MPI_FLOAT,RANK + move,0,MPI_COMM_WORLD, &request);
    //MPI_Wait(&request, &status);
}

void recieveRow(int move, vector<float> &row) {
    vector<float> fbuffer(WIDTH, 0);

    MPI_Recv(fbuffer.data(),WIDTH, MPI_FLOAT,RANK + move,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
    //MPI_Irecv(fbuffer.data(),WIDTH, MPI_FLOAT,RANK + move,MPI_ANY_TAG,MPI_COMM_WORLD,&request);
    //MPI_Wait(&request, &status);


    row =  std::move(fbuffer);
}

//helping functions

void returnFinalMap(const vector<vector<float>> &map, bool last = false) {
    vector<float> mapMessage(chunkSizes[RANK]);

    int height;
    if(last)
        height = PROCESS_HEIGHT;
    else
        height = PROCESS_HEIGHT + 1;

    int k = 0;
    for(int i = 1; i < height; i++) {
        for(int j = 0; j < WIDTH; j++) {
            mapMessage[k] = map[i][j];
            k++;
        }
    }
    MPI_Gatherv(mapMessage.data(), chunkSizes[RANK], MPI_FLOAT, nullptr, nullptr, nullptr, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
}

void prepareResult(const vector<vector<float>> &map, vector<float> &temp) {
    vector<float> tbuffer(HEIGHT * WIDTH, 0);
    MPI_Gatherv(nullptr, 0, MPI_FLOAT, tbuffer.data(), chunkSizes.data(), displ.data(), MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    int k = 0;
    for(int i = 0; i < ROOT_HEIGHT; i++) {
        for(float el: map[i]) {
            tbuffer[k] = el;
            k++;
        }
    }

//    for(int i = 0; i < WIDTH * HEIGHT; i++) {
//        if(i % WIDTH == 0) cout << endl;
//        cout << tbuffer[i] << " ";
//    }
//    cout << endl;


    temp = std::move(tbuffer);
}

//computation

float rowComputation(const vector<float> &oldRow, const vector<float> &oldRow2, vector<float> &newRow, const vector<bool> &heatSource) {
    float diff, maxDiff(0);

    if(!heatSource[0]) {
        newRow[0] = (oldRow[0] + oldRow[1] + oldRow2[0] + oldRow2[1]) / 4;

        diff = fabs(newRow[0] - oldRow[0]);
        if (diff > maxDiff)
            maxDiff = diff;
    }
    else {
        newRow[0] = oldRow[0];
    }

    for(int j = 1; j < WIDTH - 1; j++) {
        if(!heatSource[j]) {
            newRow[j] = (oldRow[j] + oldRow[j - 1] + oldRow2[j - 1] + oldRow2[j] + oldRow2[j + 1] + oldRow[j + 1]) / 6;

            diff = fabs(newRow[j] - oldRow[j]);
            if (diff > maxDiff)
                maxDiff = diff;
        }
        else {
            newRow[j] = oldRow[j];
        }
    }

    if(!heatSource[WIDTH - 1]) {
        newRow[WIDTH - 1] = (oldRow[WIDTH - 1] + oldRow[WIDTH - 2] + oldRow2[WIDTH - 1] + oldRow2[WIDTH - 2]) / 4;

        diff = fabs(newRow[WIDTH - 1] - oldRow[WIDTH - 1]);
        if (diff > maxDiff)
            maxDiff = diff;
    }
    else {
        newRow[WIDTH - 1] = oldRow[WIDTH - 1];
    }

    return maxDiff;
}

float heatMap(const vector<vector<float>> &oldMap, vector<vector<float>> &newMap, const vector<vector<bool>> &heatSource, const int height) {
    float diff, maxDiff(0);

    if(RANK == ROOT) {
        diff = rowComputation(oldMap[0], oldMap[1], newMap[0], heatSource[0]);
        if(diff > maxDiff)
            maxDiff = diff;
    }

    for(int i = 1; i < height - 1; i++) {
        for(int j = 0; j < WIDTH; j++) {
            if(!heatSource[i][j]) {
                if(j == 0) {
                    newMap[i][j] = (oldMap[i][j] + oldMap[i - 1][j] + oldMap[i - 1][j + 1] + oldMap[i][j + 1] + oldMap[i + 1][j + 1] + oldMap[i + 1][j]) / 6;
                }
                else if(j == WIDTH - 1) {
                    newMap[i][j] = (oldMap[i][j] + oldMap[i - 1][j] + oldMap[i - 1][j - 1] + oldMap[i][j - 1] + oldMap[i + 1][j - 1] + oldMap[i + 1][j]) / 6;
                }
                else {
                    newMap[i][j] = (oldMap[i][j] + oldMap[i - 1][j] + oldMap[i - 1][j + 1] + oldMap[i][j + 1] + oldMap[i + 1][j + 1] + oldMap[i + 1][j] + oldMap[i + 1][j - 1] + oldMap[i][j - 1] + oldMap[i - 1][j - 1]) / 9;
                }

                diff = fabs(newMap[i][j] - oldMap[i][j]);
                if(diff > maxDiff)
                    maxDiff = diff;

            }
            else {
                newMap[i][j] = oldMap[i][j];
            }
        }
    }

    if(RANK == WORLD_SIZE - 1) {
        int index (oldMap.size() - 1);
        diff = rowComputation(oldMap[index], oldMap[index - 1], newMap[index], heatSource[index]);
        if(diff > maxDiff)
            maxDiff = diff;
    }

    return maxDiff;
}

//iterations

bool iteration(vector<vector<float>> &map, vector<vector<float>> &newMap, vector<vector<bool>> &heatSource, vector<float> &temp, bool first = false, bool last = false) {
    //row exchanges
    if(!first)
        sendRow(-1, map[1]);

    vector<float> lowerTemp;
    if(!last)
        recieveRow(1, lowerTemp);

    if(first)
        sendRow(1, map[ROOT_HEIGHT - 1]);
    else if(!last)
        sendRow(1, map[PROCESS_HEIGHT]);

    vector<float> upperTemp;
    if(!first)
        recieveRow(-1, upperTemp);

    // computation
    float diff (-RANK);

    if(first) {
        map[ROOT_HEIGHT] = std::move(lowerTemp);
        diff = heatMap(map, newMap, heatSource, ROOT_HEIGHT + 1);
    }
    else if(last) {
        map[0] = std::move(upperTemp);
        diff = heatMap(map, newMap, heatSource, PROCESS_HEIGHT + 1);
    }
    else {
        map[0] = upperTemp;
        map[PROCESS_HEIGHT + 1] = lowerTemp;
        diff = heatMap(map, newMap, heatSource, PROCESS_HEIGHT + 2);
    }


    int sendDiff(diff > EPS_DIFF ? 1 : 0);
    vector<int> differences(WORLD_SIZE, 0);
    MPI_Allgather(&sendDiff, 1, MPI_INT, differences.data(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < WORLD_SIZE; i++) {
        if (differences[i])
            return false;
    }

    if(first)
        prepareResult(map, temp);
    else
        returnFinalMap(map);

    return true;
}

//sequential

void sequentialSolution(vector<vector<float>> &map, vector<vector<float>> &newMap, vector<vector<bool>> &heatSource, const vector<Spot> &spots, vector<float> &temp) {
    map.resize(HEIGHT, vector<float>(WIDTH, DEFAULT_TEMP));
    newMap.resize(HEIGHT, vector<float>(WIDTH, DEFAULT_TEMP));
    heatSource.resize(HEIGHT, vector<bool>(WIDTH, false));

    for (const auto &spot : spots) {
        map[spot.mX][spot.mY] = spot.mTemperature;
        newMap[spot.mX][spot.mY] = spot.mTemperature;
        heatSource[spot.mX][spot.mY] = true;
    }
    int it (0);


    while (true) {
        it++;
        ITERATION = it;

        float diff(heatMap(map, newMap, heatSource, HEIGHT));
        //cout << "It: " << ITERATION << " " << diff << endl;

        if(diff < EPS_DIFF)
            break;

        std::swap(map, newMap);
    }

    temp.resize(HEIGHT * WIDTH, 0);

    int k = 0;
    for(int i = 0; i < HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
            temp[k] = map[i][j];
            k++;
        }
    }
}

int main(int argc, char **argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int worldSize, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if (argc == 1) {
        if (myRank == 0) {
            printHelpPage(argv[0]);
        }
        MPI_Finalize();
        exit(0);
    }
    else if (argc != 3) {
        if (myRank == 0) {
            printHelpPage(argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }
    high_resolution_clock::time_point start;
    // Read the input instance.
    int width, height;  // Width and height of the matrix.
    vector<Spot> spots; // Spots with permanent temperature.
    if (myRank == 0) {
        tie(width, height, spots) = readInstance(argv[1]);
        start = high_resolution_clock::now();
    }

    //-----------------------\\
    // Insert your code here \\
    //        |  |  |        \\
    //        V  V  V        \\

    vector<vector<float>> map, newMap;
    vector<vector<bool>> heatSource;
    vector<float> temp;

    RANK = myRank;
    WORLD_SIZE = worldSize;
    WIDTH = width;
    HEIGHT = height;


    //initialization of processes

    if(WORLD_SIZE == 1) {
        sequentialSolution(map, newMap, heatSource, spots, temp);
    }
    else {
        if(RANK == ROOT) {
            PROCESS_HEIGHT = HEIGHT / WORLD_SIZE;
            ROOT_HEIGHT = PROCESS_HEIGHT + HEIGHT % WORLD_SIZE;

            rootInit(map, newMap, heatSource, spots);
        }
        else
            subProcesInit(map, newMap, heatSource);

        int it (0);
        bool stopFlag (false);
        while (!stopFlag) {
            it++;
            ITERATION = it;
            stopFlag = iteration(map, newMap, heatSource, temp, RANK == ROOT, RANK == WORLD_SIZE - 1);
            std::swap(map, newMap);
        }
    }

    // Fill this array on processor with rank 0. It must have height * width elements, and it contains the
    // linearized matrix of temperatures in row-major order
    // (see https://en.wikipedia.org/wiki/Row-_and_column-major_order)
    vector<float> temperatures (std::move(temp));

    //-----------------------\\

    if (myRank == 0) {
        double totalDuration = duration_cast<duration<double>>(high_resolution_clock::now() - start).count();
        cout << ITERATION << " computational time: " << totalDuration << " s" << endl;
        string outputFileName(argv[2]);
        writeOutput(myRank, width, height, outputFileName, temperatures);

        for(auto el: temperatures) { cout << el << " "; }
    }

    MPI_Finalize();
    return 0;
}