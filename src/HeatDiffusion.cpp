#include <mpi.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <tuple>
#include <iostream>


using namespace std;
using namespace std::chrono;

constexpr double DEFAULT_TEMP = 128;
constexpr double EPS_DIFF = 1e-5;
constexpr int ROOT = 0;

int WORLD_SIZE = -1;
int RANK = -2;

int ROOT_HEIGHT = -3;
int PROCESS_HEIGHT = -4;

int HEIGHT = -5;
int WIDTH = -6;

int ITERATION = 0;
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
    int dimensions[2] = {PROCESS_HEIGHT, WIDTH};
    MPI_Bcast(dimensions, 2, MPI_INT, ROOT, MPI_COMM_WORLD);

    vector<float> mapBcast(WORLD_SIZE * PROCESS_HEIGHT * WIDTH, 0);
    vector<int> hsBcast(WORLD_SIZE * PROCESS_HEIGHT * WIDTH, 0);

    int k = PROCESS_HEIGHT * WIDTH;
    for(int i = ROOT_HEIGHT; i < HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
            mapBcast[k] = map[i][j];
            hsBcast[k] = heatSource[i][j];
            k++;
        }
    }
    vector<float> rcv(PROCESS_HEIGHT * WIDTH, 0);

    MPI_Scatter(mapBcast.data(), PROCESS_HEIGHT * WIDTH, MPI_FLOAT, rcv.data(), PROCESS_HEIGHT * WIDTH, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Scatter(hsBcast.data(), PROCESS_HEIGHT * WIDTH, MPI_INT, rcv.data(), PROCESS_HEIGHT * WIDTH, MPI_INT, ROOT, MPI_COMM_WORLD);
}

void subProcesInit(vector<vector<float>> &map, vector<vector<float>> &newMap, vector<vector<bool>> &heatSource) {
    vector<int> ibuffer(2);
    MPI_Bcast(ibuffer.data(),ibuffer.size(), MPI_INT,ROOT,MPI_COMM_WORLD);

    PROCESS_HEIGHT = ibuffer[0];
    WIDTH = ibuffer[1];

    map.resize(PROCESS_HEIGHT, vector<float>(WIDTH, DEFAULT_TEMP));
    newMap.resize(PROCESS_HEIGHT, vector<float>(WIDTH, DEFAULT_TEMP));
    heatSource.resize(PROCESS_HEIGHT, vector<bool>(WIDTH, false));

    vector<float>matrixBuff(WIDTH * PROCESS_HEIGHT, 0);
    MPI_Scatter(nullptr, 0, MPI_FLOAT, matrixBuff.data(), PROCESS_HEIGHT * WIDTH, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    vector<int>hsBuff(WIDTH * PROCESS_HEIGHT, 0);
    MPI_Scatter(nullptr, 0, MPI_INT, hsBuff.data(), WIDTH * PROCESS_HEIGHT, MPI_INT, ROOT, MPI_COMM_WORLD);

    int k = 0;
    for(int i = 0; i < PROCESS_HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
            map[i][j] = matrixBuff[k];
            heatSource[i][j] = hsBuff[k];
            k++;
        }
    }
}

//communications
//true down, false up
void sendRow(int move, const vector<float> &row) { MPI_Send(row.data(),WIDTH, MPI_FLOAT,RANK + move,0,MPI_COMM_WORLD); }

MPI_Status status;
void recieveRow(int move, vector<float> &row) {
    vector<float> fbuffer(WIDTH, 0);

    MPI_Recv(fbuffer.data(),WIDTH, MPI_FLOAT,RANK + move,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

    row =  std::move(fbuffer);
}

//helping functions

void returnFinalMap(const vector<vector<float>> &map) {
    vector<float> mapMessage(PROCESS_HEIGHT * WIDTH);
    int k = 0;
    //cout << "P" << myRank << ": ";
    for(int i = 0; i < PROCESS_HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
            //cout << map[i][j] << " ";
            mapMessage[k] = map[i][j];
            k++;
        }
        //cout << " | ";
    }
    MPI_Gather(mapMessage.data(), WIDTH * PROCESS_HEIGHT, MPI_FLOAT, nullptr, 0, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
}

void prepareResult(const vector<vector<float>> &map, vector<float> &temp) {
    vector<float> tbuffer(WORLD_SIZE * PROCESS_HEIGHT * WIDTH, 0);
    MPI_Gather(nullptr, 0, MPI_FLOAT, tbuffer.data(), PROCESS_HEIGHT * WIDTH, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    if(PROCESS_HEIGHT == ROOT_HEIGHT) {
        //cout << "XXX" << endl;
        int k = 0;
        for (int i = 0; i < ROOT_HEIGHT; i++) {
            for (const float &el: map[i]) {
                tbuffer[k] = el;
                k++;
            }
        }
        temp = std::move(tbuffer);
    }
    else {
        temp.resize(WIDTH * HEIGHT);

        int k = 0;
        for (int i = 0; i < ROOT_HEIGHT; i++) {
            for (float el: map[i]) {
                temp[k] = el;
                k++;
            }
        }

        const int delta((ROOT_HEIGHT - PROCESS_HEIGHT) * WIDTH);

        for (int proc = 1; proc < WORLD_SIZE; proc++) {
            const int begin(proc * WIDTH * PROCESS_HEIGHT);
            const int end((proc + 1) * WIDTH * PROCESS_HEIGHT);
            //cout << "Proc:" << proc << " ";
            for (int i = begin; i < end; i++) {
                //if(i % processWidth == 0) cout << " | ";
                //cout << tbuffer[i] << " ";
                temp[i + delta] = tbuffer[i];
            }
            //cout << endl;
        }
    }
}

//computation

float middleMap(const vector<vector<float>> &oldMap, vector<vector<float>> &newMap, const vector<vector<bool>> &heatSource, const int height, const int width) {
    float diff, maxDiff(0);

    for (int i = 1; i < height - 1; i++) {
        for (int j = 0; j < width; j++) {
            if(!heatSource[i][j]) {
                if (j == 0) {
                    newMap[i][j] = (oldMap[i][j] + oldMap[i - 1][j] + oldMap[i - 1][j + 1] + oldMap[i][j + 1] + oldMap[i + 1][j + 1] + oldMap[i + 1][j]) / 6;
                }
                else if (j == width - 1){
                    newMap[i][j] = (oldMap[i][j] + oldMap[i - 1][j] + oldMap[i - 1][j - 1] + oldMap[i][j - 1] + oldMap[i + 1][j - 1] + oldMap[i + 1][j]) / 6;
                }
                else {
                    newMap[i][j] = (oldMap[i][j] + oldMap[i - 1][j - 1] + oldMap[i - 1][j] + oldMap[i - 1][j + 1] + oldMap[i][j + 1] + oldMap[i + 1][j + 1] + oldMap[i + 1][j] + oldMap[i + 1][j - 1] + oldMap[i][j - 1]) / 9;
                }
                diff = abs(newMap[i][j] - oldMap[i][j]);

                maxDiff = diff > maxDiff ? diff : maxDiff;
            }
            else {
                newMap[i][j] = oldMap[i][j];
            }
        }
    }
    return maxDiff;
}

float rootImage(const vector<vector<float>> &oldMap, vector<vector<float>> &newMap, const vector<vector<bool>> &heatSource, const vector<float> &lowerRow, const int heightLimit) {
    const int height(heightLimit);
    const int width(oldMap[0].size());

    float diff, maxDiff(0);

    if(height == 1) {
        if(!heatSource[0][0]) {
            newMap[0][0] = (oldMap[0][0] + oldMap[0][1] + lowerRow[0] + lowerRow[1]) / 4;

            diff = abs(newMap[0][0] - oldMap[0][0]);

            maxDiff = diff > maxDiff ? diff : maxDiff;
        }

        for (int j = 1; j < width - 1; j++) {
            if(!heatSource[0][j]) {
                newMap[0][j] = (oldMap[0][j] + oldMap[0][j - 1] + oldMap[1][j - 1] + lowerRow[j] + lowerRow[j + 1] + lowerRow[j + 1]) / 6;

                diff = abs(newMap[0][j] - oldMap[0][j]);

                maxDiff = diff > maxDiff ? diff : maxDiff;
            }
        }

        if(!heatSource[0][width - 1]) {
            newMap[0][width - 1] = (oldMap[0][width - 1] + oldMap[0][width - 2] + lowerRow[width - 2] + lowerRow[width - 1]) / 4;

            diff = abs(newMap[0][width - 1] - oldMap[0][width - 1]);

            maxDiff = diff > maxDiff ? diff : maxDiff;
        }
        return maxDiff;
    }

    if(!heatSource[0][0]) {
        newMap[0][0] = (oldMap[0][0] + oldMap[0][1] + oldMap[1][0] + oldMap[1][1]) / 4;

        diff = abs(newMap[0][0] - oldMap[0][0]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }

    for (int j = 1; j < width - 1; j++) {
        if(!heatSource[0][j]) {
            newMap[0][j] = (oldMap[0][j] + oldMap[0][j - 1] + oldMap[1][j - 1] + oldMap[1][j] + oldMap[1][j + 1] + oldMap[0][j + 1]) / 6;

            diff = abs(newMap[0][j] - oldMap[0][j]);

            maxDiff = diff > maxDiff ? diff : maxDiff;
        }
    }

    if(!heatSource[0][width - 1]) {
        newMap[0][width - 1] = (oldMap[0][width - 1] + oldMap[0][width - 2] + oldMap[1][width - 2] + oldMap[1][width - 1]) / 4;

        diff = abs(newMap[0][width - 1] - oldMap[0][width - 1]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }

    diff = middleMap(oldMap, newMap, heatSource, height, width);
    maxDiff = diff > maxDiff ? diff : maxDiff;

    if(!heatSource[height - 1][0]) {
        newMap[height - 1][0] = (oldMap[height - 1][0] + oldMap[height - 2][0] + oldMap[height - 2][1] + oldMap[height - 1][1] + lowerRow[0] + lowerRow[1]) / 6;

        diff = abs(newMap[height - 1][0] - oldMap[height - 1][0]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }

    for (int j = 1; j < width - 1; j++) {
        if(!heatSource[height - 1][j]) {
            newMap[height - 1][j] = (oldMap[height - 1][j] + oldMap[height - 1][j - 1] + oldMap[height - 2][j - 1] + oldMap[height - 2][j] + oldMap[height - 2][j + 1] + oldMap[height - 1][j + 1] + lowerRow[j - 1] + lowerRow[j] + lowerRow[j + 1]) / 9;

            diff = abs(newMap[height - 1][j] - oldMap[height - 1][j]);

            maxDiff = diff > maxDiff ? diff : maxDiff;
        }
    }

    if(!heatSource[height - 1][width - 1]) {
        newMap[height - 1][width - 1] = (oldMap[height - 1][width - 1] + oldMap[height - 1][width - 2] + oldMap[height - 2][width - 2] + oldMap[height - 2][width - 1] + lowerRow[width - 2] + lowerRow[width - 1]) / 6;

        diff = abs(newMap[height - 1][width - 1] - oldMap[height - 1][width - 1]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }

    return maxDiff;
}

float lastProcImage(const vector<vector<float>> &oldMap, vector<vector<float>> &newMap, const vector<vector<bool>> &heatSource, const vector<float> &upperRow) {
    const int height(oldMap.size());
    const int width(oldMap[0].size());

    float diff, maxDiff(0);

    if(height == 1) {
        if(!heatSource[height - 1][0]) {
            newMap[0][0] = (oldMap[0][0] + upperRow[0] + upperRow[1] + oldMap[0][1]) / 4;

            diff = abs(newMap[0][0] - oldMap[0][0]);

            maxDiff = diff > maxDiff ? diff : maxDiff;
        }
        else {
            newMap[0][0] = oldMap[0][0];
        }
        for (int j = 1; j < width - 1; j++) {
            if(!heatSource[0][j]) {
                newMap[0][j] = (oldMap[0][j] + oldMap[0][j - 1] + upperRow[j - 1] + upperRow[j] + upperRow[j + 1] + oldMap[0][j + 1]) / 6;

                diff = abs(newMap[0][j] - oldMap[0][j]);

                maxDiff = diff > maxDiff ? diff : maxDiff;
            }
            else {
                newMap[0][j] = oldMap[0][j];
            }
        }

        if(!heatSource[0][width - 1]) {
            newMap[0][width - 1] = (oldMap[0][width - 1] + oldMap[0][width - 2] + upperRow[width - 2] + upperRow[width - 1]) / 4;

            diff = abs(newMap[0][width - 1] - oldMap[0][width - 1]);

            maxDiff = diff > maxDiff ? diff : maxDiff;
        }
        else {
            newMap[0][width - 1] = oldMap[0][width - 1];
        }

        return maxDiff;
    }

    if(!heatSource[0][0]) {
        newMap[0][0] = (oldMap[0][0] + oldMap[0][1] + oldMap[1][0] + oldMap[1][1] + upperRow[0] + upperRow[1]) / 6;

        diff = abs(newMap[0][0] - oldMap[0][0]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }
    else {
        newMap[0][0] = oldMap[0][0];
    }

    for (int j = 1; j < width - 1; j++) {
        if(!heatSource[0][j]) {
            newMap[0][j] = (oldMap[0][j] + oldMap[0][j - 1] + oldMap[1][j - 1] + oldMap[1][j] + oldMap[1][j + 1] + oldMap[0][j + 1]+ upperRow[j - 1] + upperRow[j] + upperRow[j + 1]) / 9;

            diff = abs(newMap[0][j] - oldMap[0][j]);

            maxDiff = diff > maxDiff ? diff : maxDiff;
        }
        else {
            newMap[0][j] = oldMap[0][j];
        }
    }

    if(!heatSource[0][width - 1]) {
        newMap[0][width - 1] = (oldMap[0][width - 1] + oldMap[0][width - 2] + oldMap[1][width - 2] + oldMap[1][width - 1]+ upperRow[width - 2] + upperRow[width - 1]) / 6;

        diff = abs(newMap[0][width - 1] - oldMap[0][width - 1]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }
    else {
        newMap[0][width - 1] = oldMap[0][width - 1];
    }

    diff = middleMap(oldMap, newMap, heatSource, height, width);
    maxDiff = diff > maxDiff ? diff : maxDiff;

    if(!heatSource[height - 1][0]) {
        newMap[height - 1][0] = (oldMap[height - 1][0] + oldMap[height - 2][0] + oldMap[height - 2][1] + oldMap[height - 1][1]) / 4;

        diff = abs(newMap[height - 1][0] - oldMap[height - 1][0]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }
    else {
        newMap[height - 1][0] = oldMap[height - 1][0];
    }

    for (int j = 1; j < width - 1; j++) {
        if(!heatSource[height - 1][j]) {
            newMap[height - 1][j] = (oldMap[height - 1][j] + oldMap[height - 1][j - 1] + oldMap[height - 2][j - 1] + oldMap[height - 2][j] + oldMap[height - 2][j + 1] + oldMap[height - 1][j + 1]) / 6;

            diff = abs(newMap[height - 1][j] - oldMap[height - 1][j]);

            maxDiff = diff > maxDiff ? diff : maxDiff;
        }
        else {
            newMap[height - 1][j] = oldMap[height - 1][j];
        }
    }

    if(!heatSource[height - 1][width - 1]) {
        newMap[height - 1][width - 1] = (oldMap[height - 1][width - 1] + oldMap[height - 1][width - 2] + oldMap[height - 2][width - 2] + oldMap[height - 2][width - 1]) / 4;

        diff = abs(newMap[height - 1][width - 1] - oldMap[height - 1][width - 1]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }
    else {
        newMap[height - 1][width - 1] = oldMap[height - 1][width - 1];
    }
    return maxDiff;
}

float middleProcImage(const vector<vector<float>> &oldMap, vector<vector<float>> &newMap, const vector<vector<bool>> &heatSource, const vector<float> &upperRow, const vector<float> &lowerRow) {
    const int height(oldMap.size());
    const int width(oldMap[0].size());

    float diff, maxDiff(0);

    if(height == 1) {
        for (int j = 0; j < width; j++) {
            if(!heatSource[0][j]) {
                if (j == 0) {
                    newMap[0][j] = (oldMap[0][j] + upperRow[j] + upperRow[j + 1] + oldMap[0][j + 1] + lowerRow[j + 1] + lowerRow[j]) / 6;
                }
                else if (j == width - 1){
                    newMap[0][j] = (oldMap[0][j] + upperRow[j] + upperRow[j - 1] + oldMap[0][j - 1] + lowerRow[j - 1] + lowerRow[j]) / 6;
                }
                else {
                    newMap[0][j] = (oldMap[0][j] + upperRow[j - 1] +upperRow[j] + upperRow[j + 1] + oldMap[0][j + 1] + lowerRow[j + 1] + lowerRow[j] + lowerRow[j - 1] + oldMap[0][j - 1]) / 9;
                }
                diff = abs(newMap[0][j] - oldMap[0][j]);

                maxDiff = diff > maxDiff ? diff : maxDiff;
            }
            else {
                newMap[0][j] = oldMap[0][j];
            }
        }
        return maxDiff;
    }

    if(!heatSource[0][0]) {
        newMap[0][0] = (oldMap[0][0] + oldMap[0][1] + oldMap[1][0] + oldMap[1][1] + upperRow[0] + upperRow[1]) / 6;

        diff = abs(newMap[0][0] - oldMap[0][0]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }
    else {
        newMap[0][0] = oldMap[0][0];
    }

    for (int j = 1; j < width - 1; j++) {
        if(!heatSource[0][j]) {
            newMap[0][j] = (oldMap[0][j] + oldMap[0][j - 1] + oldMap[1][j - 1] + oldMap[1][j] + oldMap[1][j + 1] + oldMap[0][j + 1]+ upperRow[j - 1] + upperRow[j] + upperRow[j + 1]) / 9;

            diff = abs(newMap[0][j] - oldMap[0][j]);

            maxDiff = diff > maxDiff ? diff : maxDiff;
        }
        else {
            newMap[0][j] = oldMap[0][j];
        }
    }

    if(!heatSource[0][width - 1]) {
        newMap[0][width - 1] = (oldMap[0][width - 1] + oldMap[0][width - 2] + oldMap[1][width - 2] + oldMap[1][width - 1]+ upperRow[width - 2] + upperRow[width - 1]) / 6;

        diff = abs(newMap[0][width - 1] - oldMap[0][width - 1]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }
    else {
        newMap[0][width - 1] = oldMap[0][width - 1];
    }

    diff = middleMap(oldMap, newMap, heatSource, height, width);
    maxDiff = diff > maxDiff ? diff : maxDiff;

    if(!heatSource[height - 1][0]) {
        newMap[height - 1][0] = (oldMap[height - 1][0] + oldMap[height - 2][0] + oldMap[height - 2][1] + oldMap[height - 1][1] + lowerRow[0] + lowerRow[1]) / 6;

        diff = abs(newMap[height - 1][0] - oldMap[height - 1][0]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }
    else {
        newMap[height - 1][0] = oldMap[height - 1][0];
    }

    for (int j = 1; j < width - 1; j++) {
        if(!heatSource[height - 1][j]) {
            newMap[height - 1][j] = (oldMap[height - 1][j] + oldMap[height - 1][j - 1] + oldMap[height - 2][j - 1] + oldMap[height - 2][j] + oldMap[height - 2][j + 1] + oldMap[height - 1][j + 1] + lowerRow[j - 1] + lowerRow[j] + lowerRow[j + 1]) / 9;

            diff = abs(newMap[height - 1][j] - oldMap[height - 1][j]);

            maxDiff = diff > maxDiff ? diff : maxDiff;
        }
        else {
            newMap[height - 1][j] = oldMap[height - 1][j];
        }
    }

    if(!heatSource[height - 1][width - 1]) {
        newMap[height - 1][width - 1] = (oldMap[height - 1][width - 1] + oldMap[height - 1][width - 2] + oldMap[height - 2][width - 2] + oldMap[height - 2][width - 1] + lowerRow[width - 2] + lowerRow[width - 1]) / 6;

        diff = abs(newMap[height - 1][width - 1] - oldMap[height - 1][width - 1]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }
    else {
        newMap[height - 1][width - 1] = oldMap[height - 1][width - 1];
    }
    return maxDiff;
}

//iterations
int c = 10000;
bool rootIteration(const vector<vector<float>> &map, vector<vector<float>> &newMap, const vector<vector<bool>> &heatSource, vector<float> &temp) {
    //get info from lower process
    vector<float> lowerTemp;
    recieveRow(1, lowerTemp);

    //send your neighbor lower info
    sendRow(1, map[ROOT_HEIGHT - 1]);

    //computation
    float diff(-RANK);
    diff = rootImage(map, newMap, heatSource, lowerTemp, ROOT_HEIGHT);

    vector<float> sendDiff{diff};
    vector<float> differences(WORLD_SIZE, 0);
    MPI_Allgather(sendDiff.data(), 1, MPI_FLOAT, differences.data(), 1, MPI_FLOAT, MPI_COMM_WORLD);

    for (int i = 0; i < WORLD_SIZE; i++) {
        if (differences[i] > EPS_DIFF)
            return false;
    }

    //cout  << "It: " << ITERATION << endl;
    prepareResult(map, temp);
    return true;
}

bool subProcIteration(const vector<vector<float>> &map, vector<vector<float>> &newMap, const vector<vector<bool>> &heatSource, bool last = false) {
    //row exchanges
    sendRow(-1, map[0]);

    vector<float> lowerTemp;
    if(!last) {
        recieveRow(1, lowerTemp);
    }

    if(!last)
        sendRow(1, map[PROCESS_HEIGHT - 1]);

    vector<float> upperTemp;
    recieveRow(-1, upperTemp);

    // computation
    float diff (-RANK);

    if(last)
        diff = lastProcImage(map, newMap, heatSource, upperTemp);
    else
        diff = middleProcImage(map, newMap, heatSource, upperTemp, lowerTemp);

    vector<float> sendDiff{diff};
    vector<float> differences(WORLD_SIZE, 0);
    MPI_Allgather(sendDiff.data(), 1, MPI_FLOAT, differences.data(), 1, MPI_FLOAT, MPI_COMM_WORLD);

    for (int i = 0; i < WORLD_SIZE; i++) {
        if (differences[i] > EPS_DIFF)
            return false;
    }

    returnFinalMap(map);
    return true;
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

    //initialization of processes
    if(RANK == ROOT) {
        PROCESS_HEIGHT = height / WORLD_SIZE;
        ROOT_HEIGHT = PROCESS_HEIGHT + height % worldSize;
        WIDTH = width;
        HEIGHT = height;

        rootInit(map, newMap, heatSource, spots);
    }
    else
        subProcesInit(map, newMap, heatSource);


    int it (0);
    bool stopFlag (false);
    while (!stopFlag) {
        it++;
        ITERATION = it;
        if(RANK == ROOT) {
            stopFlag = rootIteration(map, newMap, heatSource, temp);
        }
        else if(RANK == WORLD_SIZE - 1) {
            stopFlag = subProcIteration(map, newMap, heatSource, true);
        }
        else {
            stopFlag = subProcIteration(map, newMap, heatSource);
        }
        std::swap(map, newMap);
    }

    // Fill this array on processor with rank 0. It must have height * width elements, and it contains the
    // linearized matrix of temperatures in row-major order
    // (see https://en.wikipedia.org/wiki/Row-_and_column-major_order)
    vector<float> temperatures = std::move(temp);

    //-----------------------\\



    if (myRank == 0) {
        double totalDuration = duration_cast<duration<double>>(high_resolution_clock::now() - start).count();
        cout << "computational time: " << totalDuration << " s" << endl;
        string outputFileName(argv[2]);
        writeOutput(myRank, width, height, outputFileName, temperatures);
        //for(auto el: temperatures) { cout << el << " "; }
    }

    MPI_Finalize();
    return 0;
}