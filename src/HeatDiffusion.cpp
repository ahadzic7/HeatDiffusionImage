#include <mpi.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <tuple>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Spot with permanent temperature on coordinates [x,y].
struct Spot {
    int mX, mY;
    float mTemperature;

    bool operator==(const Spot &b) const { return (mX == b.mX) && (mY == b.mY); }
};

tuple<int, int, vector<Spot>> readInstance(string instanceFileName) {
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

void writeOutput(const int myRank, const int width, const int height, const string outputFileName, const vector<float> &temperatures) {
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

double heatDifusionImage(const vector<vector<float>> &oldMap, vector<vector<float>> &newMap, const vector<vector<bool>> &heatSource)
{
    const int height(oldMap.size());
    const int width(oldMap[0].size());

    double diff, maxDiff(0);

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

    for (int i = 1; i < height - 1; i++) {
        for (int j = 0; j < width; j++) {
            if (j == 0 && !heatSource[i][j]) {
                newMap[i][j] = (oldMap[i][j] + oldMap[i - 1][j] + oldMap[i - 1][j + 1] + oldMap[i][j + 1] + oldMap[i + 1][j + 1] + oldMap[i + 1][j]) / 6;

                diff = abs(newMap[i][j] - oldMap[i][j]);

                maxDiff = diff > maxDiff ? diff : maxDiff;
            }
            else if (j == width - 1 && !heatSource[i][j]){
                newMap[i][j] = (oldMap[i][j] + oldMap[i - 1][j] + oldMap[i - 1][j - 1] + oldMap[i][j - 1] + oldMap[i + 1][j - 1] + oldMap[i + 1][j]) / 6;

                diff = abs(newMap[i][j] - oldMap[i][j]);

                maxDiff = diff > maxDiff ? diff : maxDiff;
            }
            else if(!heatSource[i][j]) {
                newMap[i][j] = (oldMap[i][j] + oldMap[i - 1][j - 1] + oldMap[i - 1][j] + oldMap[i - 1][j + 1] + oldMap[i][j + 1] + oldMap[i + 1][j + 1] + oldMap[i + 1][j] + oldMap[i + 1][j - 1] + oldMap[i][j - 1]) / 9;

                diff = abs(newMap[i][j] - oldMap[i][j]);

                maxDiff = diff > maxDiff ? diff : maxDiff;
            }
        }
    }

    if(!heatSource[height - 1][0]) {
        newMap[height - 1][0] = (oldMap[height - 1][0] + oldMap[height - 2][0] + oldMap[height - 2][1] + oldMap[height - 1][1]) / 4;

        diff = abs(newMap[height - 1][0] - oldMap[height - 1][0]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }
    for (int j = 1; j < width - 1; j++) {
        if(!heatSource[height - 1][j]) {
            newMap[height - 1][j] = (oldMap[height - 1][j] + oldMap[height - 1][j - 1] + oldMap[height - 2][j - 1] + oldMap[height - 2][j] + oldMap[height - 2][j + 1] + oldMap[height - 1][j + 1]) / 6;

            diff = abs(newMap[height - 1][j] - oldMap[height - 1][j]);

            maxDiff = diff > maxDiff ? diff : maxDiff;
        }
    }

    if(!heatSource[height - 1][width - 1]) {
        newMap[height - 1][width - 1] = (oldMap[height - 1][width - 1] + oldMap[height - 1][width - 2] + oldMap[height - 2][width - 2] + oldMap[height - 2][width - 1]) / 4;

        diff = abs(newMap[height - 1][width - 1] - oldMap[height - 1][width - 1]);

        maxDiff = diff > maxDiff ? diff : maxDiff;
    }
    return maxDiff;
}


void printHeatImage(const vector<vector<double>> &map) {
    for (auto row : map) {
        for (auto el : row) {
            cout << setw(4) << (int)max(min((float)el, 255.0f), 0.0f) << " ";
        }
        cout << endl;
    }
}

int main(int argc, char **argv) {

//    for(int i = 0; i < argc; i++) {
//        cout << "Arg" << i << argv[i] << endl;
//    }
//    cout << endl;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int worldSize, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

//    if (argc == 1) {
//        if (myRank == 0) {
//            printHelpPage(argv[0]);
//        }
//        MPI_Finalize();
//        exit(0);
//    } else if (argc != 3) {
//        if (myRank == 0) {
//            printHelpPage(argv[0]);
//        }
//        MPI_Finalize();
//        exit(1);
//    }

    // Read the input instance.
    int width, height;  // Width and height of the matrix.
    vector<Spot> spots; // Spots with permanent temperature.
    if (myRank == 0) {
        tie(width, height, spots) = readInstance(argv[1]);
    }

    high_resolution_clock::time_point start = high_resolution_clock::now();


    //-----------------------\\
    // Insert your code here \\
    //        |  |  |        \\
    //        V  V  V        \\

    constexpr double DEFAULT_TEMP = 0;
    constexpr double EPS_DIFF = 1e6;
    constexpr int BUFFER_SIZE = 256;
    constexpr int ROOT = 0;
    vector<float> temp;

    while (true) {
        if(myRank == ROOT) {//root process
            const int processHeight(height / worldSize);
            const int processWidth(width);
            const int rootHeight(processHeight + height % worldSize);


            vector<vector<float>> map(height, vector<float>(width, DEFAULT_TEMP));
            vector<vector<float>> newMap(height, vector<float>(width, DEFAULT_TEMP));
            vector<vector<bool>> heatSource(height, vector<bool>(width, false));

            for (auto & spot : spots) {
                map[spot.mX][spot.mY] = spot.mTemperature;
                heatSource[spot.mX][spot.mY] = true;
            }

            //bcast problem sizes to each process
            int dimensions[2] = {processHeight, width};
            MPI_Bcast(dimensions,2, MPI_INT,ROOT,MPI_COMM_WORLD);

            //prepare matrices for each process
//        cout << "ph: " << processHeight << endl;
            for(int proc = 1; proc < worldSize; proc++) {
                float *mapMessage (new float [width * processHeight]);
                int *sourceMessage (new int [width * processHeight]);
                int k = 0;
                for(int i = rootHeight + (proc - 1) * processHeight; i < rootHeight + proc * processHeight; i++) {
                    for(int j = 0; j < width; j++) {
                        mapMessage[k] = map[i][j];
                        sourceMessage[k] = heatSource[i][j];
                        k++;
                    }
                }
//            cout << "P " << proc << ": ";
//            for(int i = 0; i < width * processHeight; i++) {
//                cout << sourceMessage[i] << " ";
//            }
//            cout << endl;
                MPI_Send(mapMessage,width * processHeight, MPI_FLOAT,proc,0,MPI_COMM_WORLD);
                MPI_Send(sourceMessage,width * processHeight, MPI_INT,proc,0,MPI_COMM_WORLD);
                delete[] mapMessage;
                delete[] sourceMessage;
            }
            //starting iterations
            //send your neighbor lower info
            float *rowMessage (new float [width]);
            int row(rootHeight - 1);

            for(int j = 0; j < width; j++) {
                rowMessage[j] = map[row][j];
            }

            MPI_Send(rowMessage,width, MPI_FLOAT,myRank + 1,0,MPI_COMM_WORLD);
            delete[] rowMessage;


            //get info from lower process
            MPI_Status status;
            float fbuffer[BUFFER_SIZE];
            MPI_Recv(fbuffer,BUFFER_SIZE, MPI_FLOAT,myRank + 1,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

//        cout << "P:" << myRank << " " << "Lower: ";
            vector<float> lowerTemp(processWidth, 0);
            for(int i = 0; i < processWidth; i++) {
//            cout << fbuffer[i] << " ";
                lowerTemp[i] = fbuffer[i];
            }

            //todo: computation

            // check differences

            cout << endl;
            float diff(myRank);
            float *differences(new float [worldSize]());
            MPI_Gather(nullptr, 0, MPI_FLOAT, differences, 1, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

            differences[0] = diff;
            float maxDiff(differences[0]);
            for(int i = 0; i < worldSize; i++) {
                if(differences[i] > maxDiff)
                    maxDiff = differences[i];
                //cout << "P:" << myRank << " diff: " << differences[i] << endl;
            }
            delete[] differences;

            //send stopping info to processes
            int signal[1];
            if(maxDiff < EPS_DIFF) {
                //end iterations
                signal[0] = 1;
                MPI_Bcast(signal,1, MPI_INT,ROOT,MPI_COMM_WORLD);

                //gather results
                temp.resize(width * height);

                float *tbuffer(new float [worldSize * processWidth * processHeight]());
                MPI_Gather(nullptr, 0, MPI_FLOAT, tbuffer, processWidth * processHeight, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

                cout << "P" << myRank << ": ";

//                for(int i = 0; i < (worldSize - 1) * processWidth + rootHeight * processWidth; i++) {
//                    if(i % processWidth == 0) cout << " | ";
//                    cout << tbuffer[i] << " ";
//                }


                int k = 0;
                for(int i = 0; i < rootHeight; i++) {
                    for(float el: map[i]) {
                        temp[k] = el;
                        k++;
                    }
                }

                const int delta(rootHeight - processHeight);
                cout << endl;

                for(int proc = 1; proc < worldSize; proc++) {
                    const int begin(proc * processWidth * processHeight);
                    const int end((proc + 1) * processWidth * processHeight);
                    cout << "Proc:" << proc << " ";
                    for(int i = begin; i < end; i++) {
                        if(i % processWidth == 0) cout << " | ";
                        cout << tbuffer[i] << " ";
                        temp[i + delta * processWidth] = tbuffer[i];
                    }
                    cout << endl;
                }
                cout << endl;

                for(int i = 0; i < temp.size(); i++) {
                    if(i % processWidth == 0) cout << " | ";
                    cout << temp[i] << " ";
                }
                break;
            }
            else {
                signal[0] = 0;
                MPI_Bcast(signal,1, MPI_INT,ROOT,MPI_COMM_WORLD);
            }
        }
        else if(myRank == worldSize - 1) {//last process rowise
            //get process dimensions form root
            int ibuffer[BUFFER_SIZE];

            MPI_Status status;
            MPI_Bcast(ibuffer,2, MPI_INT,ROOT,MPI_COMM_WORLD);

            const int  processHeight(ibuffer[0]), processWidth(ibuffer[1]);
            vector<vector<float>> map(processHeight, vector<float>(processWidth, DEFAULT_TEMP));
            vector<vector<float>> newMap(processHeight, vector<float>(processWidth, DEFAULT_TEMP));
            vector<vector<bool>> heatSource(processHeight, vector<bool>(processWidth, false));

            //get process matrix
            float fbuffer[BUFFER_SIZE];

            MPI_Recv(fbuffer,BUFFER_SIZE, MPI_FLOAT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

            int receivedSize;
            MPI_Get_count(&status, MPI_FLOAT, &receivedSize);

            MPI_Recv(ibuffer,BUFFER_SIZE, MPI_INT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD, &status);

            if(receivedSize != processWidth * processHeight)
                throw std::logic_error("receivedSize != processWidth * processHeight");

//      cout << "P:" << myRank << " " << "Mat: ";
            int k = 0;
            for(int i = 0; i < processHeight; i++) {
                for(int j = 0; j < processWidth; j++) {
                    //            cout << fbuffer[k] << " ";
                    map[i][j] = fbuffer[k];
                    heatSource[i][j] = ibuffer[k];
                    k++;
                }
                //          cout << " | ";
            }

            //send info to neighbour above

            float *rowMessage (new float [processWidth]);
            //cout << "P:" << myRank << " " << "Row: ";
            for(int j = 0; j < processWidth; j++) {
                //cout << map[0][j] << " ";
                rowMessage[j] = map[0][j];
            }

            MPI_Send(rowMessage,processWidth, MPI_FLOAT,myRank - 1,0,MPI_COMM_WORLD);
            delete[] rowMessage;


            //starting iterations
            //get info from neighbour above
            MPI_Recv(fbuffer,BUFFER_SIZE, MPI_FLOAT,myRank - 1,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            MPI_Get_count(&status, MPI_FLOAT, &receivedSize);

            if(receivedSize != processWidth)
                throw std::logic_error("receivedSize != processWidth " + to_string(myRank));

            vector<float> upperTemp(processWidth, 0);

//        cout << "P:" << myRank << " " << "Upper: ";
            for(int i = 0; i < processWidth; i++) {
//            cout << fbuffer[i] << " ";
                upperTemp[i] = fbuffer[i];
            }


            //todo: computation

            //send max diff
            float diff = myRank;
            float messageDiff[1] = {diff};
            MPI_Gather(messageDiff, 1, MPI_FLOAT, nullptr, 0, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

            //stopping signal
            MPI_Bcast(ibuffer,1, MPI_INT,ROOT,MPI_COMM_WORLD);
            if(ibuffer[0] == 1) {
                float *mapMessage(new float [processWidth * processHeight]);
                k = 0;
//                cout << "P" << myRank << ": ";
                for(int i = 0; i < processHeight; i++) {
                    for(int j = 0; j < processWidth; j++) {
//                        cout << map[i][j] << " ";
                        mapMessage[k] = map[i][j];
                        k++;
                    }
//                    cout << " | ";
                }
//                cout << endl << "Size " << processWidth * processHeight << endl;
                MPI_Gather(mapMessage, processWidth * processHeight, MPI_FLOAT, nullptr, 0, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

                delete[] mapMessage;
                break;
            }
        }
        else {
            //get process dimensions form root
            int ibuffer[BUFFER_SIZE];

            MPI_Status status;
            MPI_Bcast(ibuffer,2, MPI_INT,ROOT,MPI_COMM_WORLD);

            const int processHeight (ibuffer[0]), processWidth(ibuffer[1]);
            vector<vector<float>> map(processHeight, vector<float>(processWidth, DEFAULT_TEMP));
            vector<vector<float>> newMap(processHeight, vector<float>(processWidth, DEFAULT_TEMP));
            vector<vector<bool>> heatSource(processHeight, vector<bool>(processWidth, false));

            //get process matrix
            float fbuffer[BUFFER_SIZE];

            MPI_Recv(fbuffer,BUFFER_SIZE, MPI_FLOAT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

            int receivedSize;
            MPI_Get_count(&status, MPI_FLOAT, &receivedSize);

            MPI_Recv(ibuffer,BUFFER_SIZE, MPI_INT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD, &status);

            if(receivedSize != processWidth * processHeight)
                throw std::logic_error("receivedSize != processWidth * processHeight");

            int k = 0;
//        cout << "P:" << myRank << " " << "Mat: ";
            for(int i = 0; i < processHeight; i++) {
                for(int j = 0; j < processWidth; j++) {
//                cout << fbuffer[k] << " ";
                    map[i][j] = fbuffer[k];
                    heatSource[i][j] = ibuffer[k];
                    k++;
                }
//            cout << " | ";
            }

            //send info to neighbour above and bellow

            float *rowMessageUp (new float [processWidth]), *rowMessageDown (new float [processWidth]);

            for(int j = 0; j < processWidth; j++) {
                rowMessageUp[j] = map[0][j];
                rowMessageDown[j] = map[processHeight - 1][j];
            }

            MPI_Send(rowMessageUp,processWidth, MPI_FLOAT,myRank - 1,0,MPI_COMM_WORLD);
            MPI_Send(rowMessageDown,processWidth, MPI_FLOAT,myRank + 1,0,MPI_COMM_WORLD);

            delete[] rowMessageUp;
            delete[] rowMessageDown;

            //recieve info from neighbors
            MPI_Recv(fbuffer,BUFFER_SIZE, MPI_FLOAT,myRank - 1,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

            vector<float> upperTemp(processWidth, 0);
//        cout << "P:" << myRank << " " << "Upper: ";
            for(int i = 0; i < processWidth; i++) {
                //          cout << fbuffer[i] << " ";
                upperTemp[i] = fbuffer[i];
            }
            cout << endl;

            MPI_Recv(fbuffer,BUFFER_SIZE, MPI_FLOAT,myRank + 1,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

//        cout << "P:" << myRank << " " << "Lower: ";
            vector<float> lowerTemp(processWidth, 0);
            for(int i = 0; i < processWidth; i++) {
//            cout << fbuffer[i] << " ";
                upperTemp[i] = fbuffer[i];
            }

            //todo: computation

            //send max diff to root
            float diff = myRank;
            float messageDiff[1] = {diff};
            MPI_Gather(messageDiff, 1, MPI_FLOAT, nullptr, 0, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

            //stopping signal
            MPI_Bcast(ibuffer,1, MPI_INT,ROOT,MPI_COMM_WORLD);
            if(ibuffer[0] == 1) {
                float *mapMessage(new float [processWidth * processHeight]);
                k = 0;
//                cout << "P" << myRank << ": ";
                for(int i = 0; i < processHeight; i++) {
                    for(int j = 0; j < processWidth; j++) {
//                        cout << map[i][j] << " ";
                        mapMessage[k] = map[i][j];
                        k++;
                    }
//                    cout << " | ";
                }
//                cout << endl << "Size " << processWidth * processHeight << endl;
                MPI_Gather(mapMessage, processWidth * processHeight, MPI_FLOAT, nullptr, 0, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

                delete[] mapMessage;
                break;
            }

        }
    }


    // TODO: Fill this array on processor with rank 0. It must have height * width elements and it contains the
    // linearized matrix of temperatures in row-major order
    // (see https://en.wikipedia.org/wiki/Row-_and_column-major_order)
    vector<float> temperatures = std::move(temp);

    //-----------------------\\

    double totalDuration = duration_cast<duration<double>>(high_resolution_clock::now() - start).count();
//    cout << " computational time: " << totalDuration << " s" << endl;

    if (myRank == 0) {
        string outputFileName(argv[2]);
        writeOutput(myRank, width, height, outputFileName, temperatures);
        //     cout << "Finalize p:" << myRank << endl;
        MPI_Finalize();

    }
    else {
        //   cout << "Finalize p:" << myRank << endl;
        MPI_Finalize();
    }

    return 0;
}
