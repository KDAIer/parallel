#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>
#include "mkl.h"
using namespace std;

int main()
{
    int m, k, n;
    cin >> m >> k >> n;
    vector<vector<double>> A(m, vector<double>(k));
    vector<vector<double>> B(k, vector<double>(n));
    vector<vector<double>> C(m, vector<double>(n, 0.0));
    srand(static_cast<unsigned>(time(0))); // 设置当前时间为随机种子

    for (int i = 0; i < m; ++i)
    { // 随机初始化矩阵 A
        for (int j = 0; j < k; ++j)
        {
            A[i][j] = rand() / static_cast<double>(RAND_MAX);
        }
    }

    for (int i = 0; i < k; ++i)
    { // 随机初始化矩阵 B
        for (int j = 0; j < n; ++j)
        {
            B[i][j] = rand() / static_cast<double>(RAND_MAX);
        }
    }

    auto start = chrono::high_resolution_clock::now(); // 开始时间
    // for (int i = 0; i < m; ++i)
    // { // 矩阵乘法：C = A * B
    //     for (int j = 0; j < n; ++j)
    //     {
    //         for (int l = 0; l < k; ++l)
    //         {
    //             C[i][j] += A[i][l] * B[l][j];
    //         }
    //     }
    // }
    // 列优先
    // for (int j = 0; j < n; ++j)
    // {
    //     for (int l = 0; l < k; ++l)
    //     {
    //         for (int i = 0; i < m; ++i)
    //         {
    //             C[i][j] += A[i][l] * B[l][j];
    //         }
    //     }
    // }
    // 循环展开
    // for (int i = 0; i < m; ++i)
    // {
    //     for (int j = 0; j < n; ++j)
    //     {
    //         for (int l = 0; l < k; l += 4)
    //         {
    //             C[i][j] += A[i][l] * B[l][j] + A[i][l + 1] * B[l + 1][j] + A[i][l + 2] * B[l + 2][j] + A[i][l + 3] * B[l + 3][j];
    //         }
    //     }
    // }
    auto end = chrono::high_resolution_clock::now();             // 结束时间
    chrono::duration<double> elapsed = end - start;              // 计算时间差
    cout << fixed << setprecision(6) << elapsed.count() << endl; // 输出时间差

    // 浮点性能
    double flops = 2.0 * m * k * n;                 // 浮点运算次数
    double gflops = flops / elapsed.count() / 1e9;  // 浮点性能
    cout << "浮点性能： " << gflops << " GFLOPS\n"; // 输出浮点性能
    return 0;
}
