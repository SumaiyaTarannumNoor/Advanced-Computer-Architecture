#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <iomanip>

using namespace std;

void blockedMultiply(int rowsA, int colsA, int colsB, int BS,
                     const vector<vector<double>>& A,
                     const vector<vector<double>>& B,
                     vector<vector<double>>& C)
{
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < rowsA; ii += BS) {
        for (int jj = 0; jj < colsB; jj += BS) {
            for (int kk = 0; kk < colsA; kk += BS) {

                int iMax = min(ii + BS, rowsA);
                int jMax = min(jj + BS, colsB);
                int kMax = min(kk + BS, colsA);

                for (int i = ii; i < iMax; i++) {
                    for (int k = kk; k < kMax; k++) {
                        double aik = A[i][k];
                        for (int j = jj; j < jMax; j++) {
                            C[i][j] += aik * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int N, BS, threads;

    cout << "==========================================================\n";
    cout << "   OPENMP PERFORMANCE TEST: AUTOMATIC FILLER\n";
    cout << "==========================================================\n\n";

    cout << "Enter Matrix Size: ";
    cin >> N;

    cout << "Enter Block Size (e.g., 32): "; cin >> BS;
    cout << "Enter Number of Threads (1, 2, or 4): "; cin >> threads;

    // Auto-fill matrices so you don't have to type
    vector<vector<double>> A(N, vector<double>(N, 1.0));
    vector<vector<double>> B(N, vector<double>(N, 2.0));
    vector<vector<double>> C(N, vector<double>(N, 0.0));

    omp_set_num_threads(threads);

    cout << "\nCalculating... Please wait." << endl;

    double start = omp_get_wtime();
    blockedMultiply(N, N, N, BS, A, B, C);
    double end = omp_get_wtime();

    cout << "\n----------------------------------------------------------\n";
    cout << " PERFORMANCE RESULTS\n";
    cout << "----------------------------------------------------------\n";
    cout << "Matrix Size: " << N << " x " << N << endl;
    cout << "Threads    : " << threads << endl;
    cout << "Block Size : " << BS << endl;
    cout << "Time       : " << fixed << setprecision(8) << (end - start) << " seconds" << endl;
    cout << "----------------------------------------------------------\n";

    // If size is 32 or less, we print the result to verify it's working
    if (N <= 32) {
        cout << "\nResult Matrix (First 5x5 shown for brevity):\n";
        int limit = min(N, 5);
        for(int i = 0; i < limit; i++) {
            for(int j = 0; j < limit; j++) {
                cout << setw(8) << C[i][j] << " ";
            }
            cout << endl;
        }
    }

    cout << "\nPress Enter to close...";
    cin.ignore(1000, '\n'); cin.get();
    return 0;
}
