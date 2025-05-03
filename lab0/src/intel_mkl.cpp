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

    // 使用一维数组，避免 std::vector 的二维数据存储问题
    vector<double> A(m * k);      // 存储 A
    vector<double> B(k * n);      // 存储 B
    vector<double> C(m * n, 0.0); // 存储 C，初始化为0

    srand(static_cast<unsigned>(time(0))); // 设置当前时间为随机种子

    // 随机初始化矩阵 A
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            A[i * k + j] = rand() / static_cast<double>(RAND_MAX);
        }
    }

    // 随机初始化矩阵 B
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            B[i * n + j] = rand() / static_cast<double>(RAND_MAX);
        }
    }

    auto start = chrono::high_resolution_clock::now();
    // MKL
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A.data(), k, B.data(), n, 0.0, C.data(), n);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << fixed << setprecision(6) << elapsed.count() << endl;

    // 浮点性能
    double flops = 2.0 * m * k * n; // 浮点运算次数
    double gflops = flops / elapsed.count() / 1e9;
    cout << "浮点性能： " << gflops << " GFLOPS\n";
    return 0;
}
