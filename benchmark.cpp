#include <emp-tool/emp-tool.h>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
#include <gmpxx.h>
using namespace emp;


std::string key[101][101];
PRG k_P("1111111111111111");
std::vector<std::vector<PRG>> k_ij;
uint64_t COMM = 0;
uint64_t LAN = 1;

std::string generateRandomString()
{
    static const char charset[] =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_int_distribution<int> dist(0, sizeof(charset) - 2);

    std::string randomString;
    for (int i = 0; i < 16; ++i) {
        randomString += charset[dist(rng)];
    }
    return randomString;
}

void generateSymmetricMatrix(int n)
{
    for (int i = 0; i <= n; ++i) {
        for (int j = i; j <= n; ++j)
        {
            key[i][j] = generateRandomString();
            key[j][i] = key[i][j];
        }
    }
}


void printMatrix(int n)
{
    for (int i = 0; i <= n; ++i)
    {
        for (int j = 0; j <= n; ++j)
        {
            std::cout << key[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void initialize_PRG(size_t n)
{
    generateSymmetricMatrix(n);
    k_ij.resize(n+1, std::vector<PRG>(n+1));

    for (size_t i = 0; i <= n; i++)
    {
        for (size_t j = 0; j <= n; j++)
        {
            k_ij[i][j].reseed((const block*)key[i][j].c_str(), 0);
            // int a;
            // k_ij[i][j].random_data(&a, 4);
            // std::cout << a << " ";
        }
        // std::cout << std::endl;
    }
}

void SS_SH(std::vector<uint64_t>& inputs, int n)
{
    // std::cout << a << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint64_t> xi(n);
    int size = inputs.size();
    uint64_t* arr = new uint64_t[size];
    for (int i = 0; i < n; i++)
    {
        if (i == 0)
        {
            continue;
        }
        k_ij[0][i].random_data(arr, sizeof(uint64_t)*size);
        /* 
        for (int k = 0; k < size; k++)
        {
            std::cout << arr[k] << " ";
        }
        std::cout << "\n";
        */
        for (int j = 0; j < size; j++)
        {
            inputs[j] -= arr[j];
        }
    }
    delete[] arr;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Secret Sharing execution time: " << duration << " microseconds" << std::endl;
    /*
    for (int i = 0; i < size; i++)
    {
        std::cout << inputs[i] << " ";
    }
    std::cout << "\n";
    */
}

void BT_SH(int n, int size)
{
    //auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t* a = new uint64_t[size];
    uint64_t* b = new uint64_t[size];
    uint64_t* ab = new uint64_t[size];

    uint64_t* a_temp = new uint64_t[size];
    uint64_t* b_temp = new uint64_t[size];
    uint64_t* ab_temp = new uint64_t[size];
    
    k_ij[n][n].random_data(a, sizeof(uint64_t)*size);
    k_ij[n][n].random_data(b, sizeof(uint64_t)*size);
    for (int i = 0; i < size; i++)
    {
        ab[i] = a[i] * b[i];
    }

    for (int i = 0; i < n-1; i++)
    {
        k_ij[i][n].random_data(a_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(b_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(ab_temp, sizeof(uint64_t)*size);
        /*
        std::cout << i << ":\n";
        for (int j = 0; j < size; j++)
        {
            std::cout << "("<< a_temp[j] << ", " << b_temp[j] << ", " << ab_temp[j]<< ") ";
        }
        std::cout << "\n";
        */
        for (int j = 0; j < size; j++)
        {
            a[j] -= a_temp[j];
            b[j] -= b_temp[j];
            ab[j] -= ab_temp[j];
        }
    }
    /*
    std::cout << n-1 << ":\n";
    for (int j = 0; j < size; j++)
    {
        std::cout << "("<< a[j] << ", " << b[j] << ", " << ab[j]<< ") ";
    }
    std::cout << "\n";
    */

    /* simulate network cost */
    uint64_t comm = 64 * 3  * size;
    COMM += comm;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }


    delete[] a;
    delete[] b;
    delete[] ab;
    delete[] a_temp;
    delete[] b_temp;
    delete[] ab_temp;

    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "Generate Beaver triples execution time: " << duration << " microseconds" << std::endl;
}

uint64_t Mul_SH(uint64_t xi, uint64_t yi, uint64_t ai, uint64_t bi, uint64_t abi,
        std::vector<uint64_t>& xi_ai, std::vector<uint64_t>& yi_bi, int party, int n)
{
    // auto start = std::chrono::high_resolution_clock::now();

    uint64_t x_a = xi - ai;
    uint64_t y_b = yi - bi;
    /* simulate network cost */
    uint64_t comm = 64 * 2 * (n-1);
    COMM += comm * n;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }


    for (int i = 0; i < xi_ai.size(); i++)
    {
        x_a += xi_ai[i];
        y_b += yi_bi[i];
    }
    uint64_t xy = ai * y_b + bi * x_a + abi;
    if (party == 0)
    {
        xy = xy + x_a * y_b;
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // std::cout << "Multiplication execution time: " << duration << " microseconds" << std::endl;
    return xy;
}

void GAD_SH(int n, int size, int l, int m)
{
    uint64_t mask1 = (1ULL << l) - 1;
    uint64_t mask2 = (1ULL << (l-m)) - 1;
    //std::cout << "mask1: " << std::hex << mask1 << std::endl;
    //std::cout << "mask2: " << std::hex << mask2 << std::endl;
    //auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t* r = new uint64_t[size];
    uint64_t* r1 = new uint64_t[size];
    uint64_t* b = new uint64_t[size];

    uint64_t* r_temp = new uint64_t[size];
    uint64_t* r1_temp = new uint64_t[size];
    uint64_t* b_temp = new uint64_t[size];
    
    k_ij[n][n].random_data(r, sizeof(uint64_t)*size);
    k_ij[n][n].random_data(r1, sizeof(uint64_t)*size);
    k_ij[n][n].random_data(b, sizeof(uint64_t)*size);
    for (int i = 0; i < size; i++)
    {
        //std::cout << "r: " << std::hex << r[i] << std::endl;
        //std::cout << "r1: " << std::hex << r1[i] << std::endl;
        //std::cout << "b: " << std::hex << b[i] << std::endl;
        r[i]  = r[i] & mask1;
        r1[i] = r1[i] & mask2;
        b[i] = b[i] & 1;
        //std::cout << "r: " << std::hex << r[i] << std::endl;
        //std::cout << "r1: " << std::hex << r1[i] << std::endl;
        //std::cout << "b: " << std::hex << b[i] << std::endl;
    }

    for (int i = 0; i < n-1; i++)
    {
        k_ij[i][n].random_data(r_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(r1_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(b_temp, sizeof(uint64_t)*size);
        
        for (int j = 0; j < size; j++)
        {
            r[j]  -= r_temp[j];
            r1[j] -= r1_temp[j];
            b[j]  -= b_temp[j];
        }

        /*
        std::cout << i << ":\n";
        for (int j = 0; j < size; j++)
        {
            std::cout << "("<< a_temp[j] << ", " << b_temp[j] << ", " << ab_temp[j]<< ") ";
        }
        std::cout << "\n";
        */
    }
    /* simulate network cost */
    uint64_t comm = 64 * 3 * size;
    COMM += comm;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }
    
    /*
    std::cout << n-1 << ":\n";
    for (int j = 0; j < size; j++)
    {
        std::cout << "("<< a[j] << ", " << b[j] << ", " << ab[j]<< ") ";
    }
    std::cout << "\n";
    */

    delete[] r;
    delete[] r1;
    delete[] b;
    delete[] r_temp;
    delete[] r1_temp;
    delete[] b_temp;

    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "Generate truncation auxiliary data execution time: " << duration << " microseconds" << std::endl;

}


uint64_t Trunc_SH(uint64_t xi, uint64_t ri, uint64_t r1i, uint64_t bi,
         int party, int l, int m, int n)
{
    //auto start = std::chrono::high_resolution_clock::now();
    uint64_t k = 64;
    uint64_t l2 = (1ULL << l);
    uint64_t m2 = (1ULL << m);
    uint64_t kl = (1ULL << (64-l-1));
    uint64_t c = kl * (xi + l2 * bi + m2 * ri + r1i);
    /* simulate network cost */
    uint64_t comm = 64 * (n-1);
    COMM += comm * n;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }

    std::vector<uint64_t> ci(n-1);
    for (int i = 0; i < ci.size(); i++)
    {
        c += ci[i];
    }
    uint64_t c1 = c >> (k-l-1);
    uint64_t cl = (c >> l) & 1;
    uint64_t v = bi - 2 * cl * bi;
    if (party == 1)
    {
        v += cl;
    }

    uint64_t l_m = (1ULL << l - m);
    uint64_t result = -ri + (1ULL << l - m) * v;
    if (party == 1)
    {
        c1 = c1 % (1ULL << l);
        c1 = c1 >> m;
        result += c1;
    }
    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "SH Truncation execution time: " << duration << " microseconds" << std::endl;
    return result;
}

void GTEZ_SH(uint64_t xi, int lx, std::vector<uint64_t>& ri, int party,
        std::vector<uint64_t>& r1i, std::vector<uint64_t>& bi, std::vector<std::vector<uint64_t>>& ci,
        std::vector<std::vector<uint64_t>>& w, int n)
{
    //auto start = std::chrono::high_resolution_clock::now();
    // {r0, ...rlx, r*, s, t}
    uint64_t* rands = new uint64_t[lx+4];
    k_P.random_data(rands, sizeof(uint64_t)*(lx+4));
    uint64_t* rands2 = new uint64_t[n-1];
    k_P.random_data(rands2, sizeof(uint64_t)*(n-1));
    
    // t
    uint64_t t = rands[lx+3] & 1;
    if (t == 1)
    {
        xi = xi * UINT64_MAX;
    }
    // u_*
    uint64_t ux = UINT64_MAX;
    for (int i = 0; i < n-1; i++)
    {
        ux -= rands2[i];
    }
    // v_*
    uint64_t vx = ux + 3 * xi;
    if (party == 1)
    {
        vx -= 1;
    }
    // array c
    uint64_t l2 = (1ULL << lx);
    uint64_t kl = (1ULL << (64-lx-1));
    uint64_t* arr_c = new uint64_t[lx];
    for (int i = 0; i < lx; i++)
    {
        uint64_t m2 = (1ULL << i+1);
        arr_c[i] = kl * (xi + l2 * bi[i] + m2 * ri[i] + r1i[i]);
    }

    /* reconstruct c, simulate network cost */
    uint64_t comm = 64 * (n-1) * lx;
    COMM += comm * n;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }
    
    for (int i = 0; i < ci.size(); i++)
    {
        for (int j = 0; j < lx; j++)
        {
            arr_c[j] += ci[i][j];
        }
    }

    // compute truncation of  x
    uint64_t* c1 = new uint64_t[lx];
    uint64_t* cl = new uint64_t[lx];
    uint64_t* v = new uint64_t[lx];
    uint64_t k = 64;
    for (int i = 0; i < lx; i++)
    {
        c1[i] = arr_c[i] >> (k - lx - 1);
        cl[i] = (arr_c[i] >> lx) & 1;
        v[i] = bi[i] - 2 * cl[i] * bi[i];
        if (party == 1)
        {
            v[i] += cl[i];
        }
    }
    std::vector<uint64_t> wi;
    wi.resize(lx + 2);
    wi[0] = xi;
    for (int i = 1; i <= lx; i++)
    {
        uint64_t m = i;
        uint64_t l_m = (1ULL << lx - m);
        wi[i] = -ri[i-1] + (1ULL << lx - m) * v[i-1];
        if (party == 1)
        {
            c1[i-1] = c1[i-1] % (1ULL << lx);
            c1[i-1] = c1[i-1] >> m;
            wi[i] += c1[i-1];
        }
    }
    
    for (int i = lx-1; i >= 0; i--)
    {
        wi[i] += wi[i+1];
        if (party == 1)
        {
            wi[i] -= 1;
        }
        wi[i] *= rands[i];
    }
    if (party == 1)
    {
        wi[lx] -= 1;
    }
    wi[lx] *= rands[lx];
    wi[lx+1] = vx * rands[lx+1];
    // shuffle
    std::mt19937 rng(rands[lx+2]);
    std::shuffle(wi.begin(), wi.end(), rng);

    /* Pn+1 reconstruct wi, simulate network cost */
    comm = 64 * (lx + 2);
    COMM += comm * n;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }

    uint64_t r = 0;
    for (int i = 0; i < lx+2; i++)
    {
        uint64_t temp = 0;
        for (int j = 0; j < n; j++)
        {
            temp += w[j][i];
        }
        if (temp == 0)
        {
            r = 1;
            break;
        }
    }
    for (int i = 0; i < n-1; i++)
    {
        uint64_t random;
        k_ij[i][n].random_data(&random, sizeof(uint64_t));
        r -= random;
    }

    // Pn+1 send [GTEZ(x)] to Pn, simulate network cost
    comm = 64;
    COMM += comm;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }
    
    // Pi compute GTEZ(x)
    uint64_t rst = r - 2 * t * r;
    if (party == 1)
    {
        rst += t;
    }
    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "Greater than or equal to zero execution time: " << duration << " microseconds" << std::endl;
    
    delete[] rands;
    delete[] rands2;
    delete[] arr_c;
    delete[] c1;
    delete[] cl;
    delete[] v;
}

uint64_t modInverse(uint64_t a_val)
{
    mpz_t a, m, inv;
    mpz_init(a);
    mpz_init(m);
    mpz_init(inv);

    mpz_set_ui(a, a_val);
    mpz_ui_pow_ui(m, 2, 64);    // m = 2^64

    if (mpz_invert(inv, a, m) != 0)
    {
        uint64_t inv_val = mpz_get_ui(inv);
        return inv_val;
    }
    return 0;
}

void MAC_SH(int n)
{
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t delta;
    k_ij[n][n].random_data(&delta, sizeof(uint64_t)*1);
    delta |= 1;
    uint64_t delta_1 = modInverse(delta);
    while (!delta_1)
    {
        k_ij[n][n].random_data(&delta, sizeof(uint64_t)*1);
        delta |= 1;
        delta_1 = modInverse(delta);
    }
    std::cout << "delta: " << delta << " delta^-1: " << delta_1 << "\n";
    uint64_t delta_i[2];
    for (int i = 0; i < n-1; i++)
    {
        k_ij[i][n].random_data(delta_i, sizeof(uint64_t)*2);
        delta -= delta_i[0];
        delta_1 -= delta_i[1];
    }
    std::cout << "delta: " << delta << " delta^-1: " << delta_1 << "\n";
    // Pn+1 send [delta, delta_1] to Pn, simulate network cost
    uint64_t comm = 64 * 2;
    COMM += comm;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Generate Mac Key execution time: " << duration << " microseconds" << std::endl;
}

void challenge_Data_SH(int size, int n)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t* v = new uint64_t[size];
    uint64_t* v_temp = new uint64_t[size];
    k_ij[n][n].random_data(v, sizeof(uint64_t)*size);
    

    for (int i = 0; i < n-1; i++)
    {
        k_ij[i][n].random_data(v_temp, sizeof(uint64_t)*size);
        /*
        std::cout << i << ":\n";
        for (int j = 0; j < size; j++)
        {
            std::cout << "("<< a_temp[j] << ", " << b_temp[j] << ", " << ab_temp[j]<< ") ";
        }
        std::cout << "\n";
        */
        for (int j = 0; j < size; j++)
        {
            v[j] -= v_temp[j];
        }
    }
    /*
    std::cout << n-1 << ":\n";
    for (int j = 0; j < size; j++)
    {
        std::cout << "("<< a[j] << ", " << b[j] << ", " << ab[j]<< ") ";
    }
    std::cout << "\n";
    */

    /* simulate network cost */
    uint64_t comm = 64 * size;
    COMM += comm;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }

    delete[] v;
    delete[] v_temp;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Generate Challenge Data execution time: " << duration << " microseconds" << std::endl;
}

void Verify_DH(std::vector<uint64_t> xi, std::vector<uint64_t> d_xi,
        uint64_t vi, uint64_t deltai, uint64_t v, uint64_t delta, int n)
{
    //auto start = std::chrono::high_resolution_clock::now();
    uint64_t sum = 0;
    uint64_t sum2 = 0;
    for (int i = 0; i < xi.size(); i++)
    {
        sum += xi[i];
        sum2 += d_xi[i];
    }

    sum += vi;
    /* reconstruct x + v, simulate network cost */
    uint64_t comm = 64;
    COMM += comm * n;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }

    std::vector<uint64_t> x_v(n-1);
    for (int i = 0; i < n-1; i++)
    {
        sum += x_v[i];
    }


    uint64_t ti = delta * sum - sum2;
    /* send ti to Pn+1, simulate network cost */
    comm = 64;
    COMM += comm * n;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }
    std::vector<uint64_t> t(n);
    uint64_t result = v * delta;
    for (int i = 0; i < t.size(); i++)
    {
        result -= t[i];
    }
    result = (result == 0 ? 1 : 0);
    /* send result to P1-Pn, simulate network cost */
    comm = 64 * n;
    COMM += comm;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }
    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "Verify Data execution time: " << duration << " microseconds" << std::endl;
}


void SS_SH2(std::vector<uint64_t>& x, uint64_t deltai, uint64_t delta, int party, int n)
{
    // std::cout << a << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    int size = x.size();
    uint64_t* t = new uint64_t[size];
    uint64_t* r = new uint64_t[size];
    k_ij[party-1][n].random_data(t, sizeof(uint64_t)*size);
    k_P.random_data(r, sizeof(uint64_t)*size);

    for (int i = 0; i < size; i++)
    {
        x[i] = x[i] + r[i] + t[i];
    }

    /* send x + r + t to Pn+1, simulate network cost */
    uint64_t comm = 64 * size;
    COMM += comm;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }

    std::vector<uint64_t> d_x(size);
    for (int i = 0; i < size; i++)
    {
        x[i] = x[i] - t[i];
        d_x[i] = x[i] * delta;
    }
    
    uint64_t* rand = new uint64_t[size*2];
    for (int i = 0; i < n-1; i++)
    {
        k_ij[i][n].random_data(rand, sizeof(uint64_t)*size*2);
        
        for (int j = 0; j < size; j++)
        {
            x[j] -= rand[j];
        }
        for (int j = size; j < 2*size; j++)
        {
            d_x[j-size] -= rand[j];
        }
    }

    /* send {deltax_n, x_n} to Pn, simulate network cost */
    comm = 64 * size * 2;
    COMM += comm;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }

    for (int i = 0; i < size; i++)
    {
        d_x[i] -= deltai * r[i];
    }

    if (party)
    {
        for (int i = 0; i < size; i++)
        {
            x[i] -= r[i];
        }
    }
    delete[] t;
    delete[] r;
    delete[] rand;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Secret Sharing execution time: " << duration << " microseconds" << std::endl;
    /*
    for (int i = 0; i < size; i++)
    {
        std::cout << inputs[i] << " ";
    }
    std::cout << "\n";
    */
}

void BT_SH2(int n, int size)
{
    //auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t* a = new uint64_t[size];
    uint64_t* b = new uint64_t[size];
    uint64_t* ab = new uint64_t[size];
    uint64_t* a1 = new uint64_t[size];
    uint64_t* b1 = new uint64_t[size];
    uint64_t* c1 = new uint64_t[size];
    uint64_t* ab1 = new uint64_t[size];
    uint64_t* ac1 = new uint64_t[size];
    uint64_t* bc1 = new uint64_t[size];
    uint64_t* abc1 = new uint64_t[size];


    uint64_t* a_temp = new uint64_t[size];
    uint64_t* b_temp = new uint64_t[size];
    uint64_t* ab_temp = new uint64_t[size];
    uint64_t* a1_temp = new uint64_t[size];
    uint64_t* b1_temp = new uint64_t[size];
    uint64_t* c1_temp = new uint64_t[size];
    uint64_t* ab1_temp = new uint64_t[size];
    uint64_t* ac1_temp = new uint64_t[size];
    uint64_t* bc1_temp = new uint64_t[size];
    uint64_t* abc1_temp = new uint64_t[size];
    
    k_ij[n][n].random_data(a, sizeof(uint64_t)*size);
    k_ij[n][n].random_data(b, sizeof(uint64_t)*size);
    k_ij[n][n].random_data(a1, sizeof(uint64_t)*size);
    k_ij[n][n].random_data(b1, sizeof(uint64_t)*size);
    k_ij[n][n].random_data(c1, sizeof(uint64_t)*size);
    for (int i = 0; i < size; i++)
    {
        ab[i] = a[i] * b[i];
        ab1[i] = a1[i] * b1[i];
        ac1[i] = a1[i] * c1[i];
        bc1[i] = b1[i] * c1[i];
        abc1[i] = a1[i] * b1[i] * c1[i];

    }

    std::cout << a[0] << " " << b[0] << " " << ab[0] << "\n";
    std::cout << a1[0] << " " << b1[0] << " " << c1[0] << " " << ab1[0] << " " << ac1[0] << " " << bc1[0] << " " << abc1[0] << "\n";


    for (int i = 0; i < n-1; i++)
    {
        k_ij[i][n].random_data(a_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(b_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(ab_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(a1_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(b1_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(c1_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(ab1_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(ac1_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(bc1_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(abc1_temp, sizeof(uint64_t)*size);
        /*
        std::cout << i << ":\n";
        for (int j = 0; j < size; j++)
        {
            std::cout << "("<< a_temp[j] << ", " << b_temp[j] << ", " << ab_temp[j]<< ") ";
        }
        std::cout << "\n";
        */
        for (int j = 0; j < size; j++)
        {
            a[j] -= a_temp[j];
            b[j] -= b_temp[j];
            ab[j] -= ab_temp[j];
            a1[j] -= a1_temp[j];
            b1[j] -= b1_temp[j];
            c1[j] -= c1_temp[j];
            ab1[j] -= ab1_temp[j];
            ac1[j] -= ac1_temp[j];
            bc1[j] -= bc1_temp[j];
            abc1[j] -= abc1_temp[j];
        }
    }
    /*
    std::cout << n-1 << ":\n";
    for (int j = 0; j < size; j++)
    {
        std::cout << "("<< a[j] << ", " << b[j] << ", " << ab[j]<< ") ";
    }
    std::cout << "\n";
    */

    /* simulate network cost */
    uint64_t comm = 64 * 10 * size;
    COMM += comm;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }

    delete[] a;
    delete[] b;
    delete[] ab;
    delete[] a1;
    delete[] b1;
    delete[] c1;
    delete[] ab1;
    delete[] ac1;
    delete[] abc1;

    delete[] a_temp;
    delete[] b_temp;
    delete[] ab_temp;
    delete[] a1_temp;
    delete[] b1_temp;
    delete[] c1_temp;
    delete[] ab1_temp;
    delete[] ac1_temp;
    delete[] bc1_temp;
    delete[] abc1_temp;

    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "Generate Beaver triples DH execution time: " << duration << " microseconds" << std::endl;
}

void Mul_DH(uint64_t xi, uint64_t yi, uint64_t deltaxi,
        uint64_t deltayi, uint64_t delta_1, uint64_t ai, uint64_t bi,
        uint64_t abi, uint64_t a1, uint64_t b1, uint64_t c1, uint64_t ab1,
        uint64_t ac1, uint64_t bc1, uint64_t abc1, int party, int n)
{
    // auto start = std::chrono::high_resolution_clock::now();

    uint64_t x_a = xi - ai;
    uint64_t y_b = yi - bi;
    uint64_t dx_a = deltaxi - a1;
    uint64_t dy_b = deltayi - b1;
    uint64_t d_c = delta_1 - c1;
    /* simulate network cost */
    uint64_t comm = 64 * 5 * (n-1);
    COMM += comm * n;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }
    std::vector<uint64_t> xi_ai(n-1);
    std::vector<uint64_t> yi_bi(n-1);
    std::vector<uint64_t> dxi_ai(n-1);
    std::vector<uint64_t> dyi_bi(n-1);
    std::vector<uint64_t> di_ci(n-1);

    for (int i = 0; i < xi_ai.size(); i++)
    {
        x_a += xi_ai[i];
        y_b += yi_bi[i];
        dx_a += dxi_ai[i];
        dy_b += dyi_bi[i];
        d_c += di_ci[i];
    }
    uint64_t xy = ai * y_b + bi * x_a + abi;
    uint64_t dxy = c1 * dx_a * dy_b + a1 * dy_b * d_c + b1 * dx_a * d_c + ac1 * dy_b + bc1 * dx_a + ab1 * d_c + abc1;
    if (party == 1)
    {
        xy = xy + x_a * y_b;
        dxy = dxy + dx_a * dy_b * d_c;
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // std::cout << "Multiplication execution time: " << duration << " microseconds" << std::endl;
}

void GAD_SH2(uint64_t delta, int n, int size, int l, int m)
{
    uint64_t mask1 = (1ULL << l) - 1;
    uint64_t mask2 = (1ULL << (l-m)) - 1;
    //std::cout << "mask1: " << std::hex << mask1 << std::endl;
    //std::cout << "mask2: " << std::hex << mask2 << std::endl;
    //auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t* r = new uint64_t[size];
    uint64_t* r1 = new uint64_t[size];
    uint64_t* b = new uint64_t[size];
    uint64_t* d_r = new uint64_t[size];
    uint64_t* d_r1 = new uint64_t[size];
    uint64_t* d_b = new uint64_t[size];

    uint64_t* r_temp = new uint64_t[size];
    uint64_t* r1_temp = new uint64_t[size];
    uint64_t* b_temp = new uint64_t[size];
    uint64_t* dr_temp = new uint64_t[size];
    uint64_t* dr1_temp = new uint64_t[size];
    uint64_t* db_temp = new uint64_t[size];
    
    k_ij[n][n].random_data(r, sizeof(uint64_t)*size);
    k_ij[n][n].random_data(r1, sizeof(uint64_t)*size);
    k_ij[n][n].random_data(b, sizeof(uint64_t)*size);
    for (int i = 0; i < size; i++)
    {
        //std::cout << "r: " << std::hex << r[i] << std::endl;
        //std::cout << "r1: " << std::hex << r1[i] << std::endl;
        //std::cout << "b: " << std::hex << b[i] << std::endl;
        r[i]  = r[i] & mask1;
        r1[i] = r1[i] & mask2;
        b[i] = b[i] & 1;
        d_r[i] = delta * r[i];
        d_r1[i] = delta * r1[i];
        d_b[i] = delta * b[i];
        //std::cout << "r: " << std::hex << r[i] << std::endl;
        //std::cout << "r1: " << std::hex << r1[i] << std::endl;
        //std::cout << "b: " << std::hex << b[i] << std::endl;
    }

    for (int i = 0; i < n-1; i++)
    {
        k_ij[i][n].random_data(r_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(r1_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(b_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(dr_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(dr1_temp, sizeof(uint64_t)*size);
        k_ij[i][n].random_data(db_temp, sizeof(uint64_t)*size);
        
        for (int j = 0; j < size; j++)
        {
            r[j]  -= r_temp[j];
            r1[j] -= r1_temp[j];
            b[j]  -= b_temp[j];
            d_r[j] -= dr_temp[j];
            d_r1[j] -= dr1_temp[j];
            d_b[j] -= db_temp[j];
        }

        /*
        std::cout << i << ":\n";
        for (int j = 0; j < size; j++)
        {
            std::cout << "("<< a_temp[j] << ", " << b_temp[j] << ", " << ab_temp[j]<< ") ";
        }
        std::cout << "\n";
        */
    }
    /* simulate network cost */
    uint64_t comm = 64 * 6 * size;
    COMM += comm;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }
    
    /*
    std::cout << n-1 << ":\n";
    for (int j = 0; j < size; j++)
    {
        std::cout << "("<< a[j] << ", " << b[j] << ", " << ab[j]<< ") ";
    }
    std::cout << "\n";
    */

    delete[] r;
    delete[] r1;
    delete[] b;
    delete[] d_r;
    delete[] d_r1;
    delete[] d_b;
    delete[] r_temp;
    delete[] r1_temp;
    delete[] b_temp;
    delete[] dr_temp;
    delete[] dr1_temp;
    delete[] db_temp;

    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "DH Generate truncation auxiliary data execution time: " << duration << " microseconds" << std::endl;

}

uint64_t Trunc_DH(uint64_t xi, uint64_t ri, uint64_t r1i, uint64_t bi,
        uint64_t d_xi, uint64_t d_ri, uint64_t d_r1i, uint64_t d_bi, 
        uint64_t deltai, int party, int l, int m, int n)
{
    //auto start = std::chrono::high_resolution_clock::now();
    uint64_t k = 64;
    uint64_t l2 = (1ULL << l);
    uint64_t m2 = (1ULL << m);
    uint64_t kl = (1ULL << (64-l-1));
    uint64_t c = kl * (xi + l2 * bi + m2 * ri + r1i);
    uint64_t d_c = kl * (d_xi + l2 * d_bi + m2 * d_ri + d_r1i);
    
    /* reconstruct c, simulate network cost */
    uint64_t comm = 64 * (n-1);
    COMM += comm * n;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }
    
    std::vector<uint64_t> ci(n-1);
    for (int i = 0; i < ci.size(); i++)
    {
        c += ci[i];
    }

    // verify c
    std::vector<uint64_t> arr_xi = {c};
    std::vector<uint64_t> arr_d_xi = {d_c};
    uint64_t vi, v1, delta;
    Verify_DH(arr_xi, arr_d_xi, vi, deltai, v1, delta, n);


    uint64_t c1 = c >> (k-l-1);
    uint64_t cl = (c >> l) & 1;
    uint64_t d_cl = deltai * cl;
    uint64_t v = bi - 2 * cl * bi;
    if (party == 1)
    {
        v += cl;
    }
    uint64_t d_v = d_bi + d_cl - 2 * cl * d_bi;

    uint64_t l_m = (1ULL << l - m);
    uint64_t result = -ri + (1ULL << l - m) * v;
    c1 = c1 % (1ULL << l);
    c1 = c1 >> m;
    uint64_t d_result = deltai * c1 - d_ri + + (1ULL << l - m) * d_v;
    if (party == 1)
    {
        result += c1;
    }
    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "DH Truncation execution time: " << duration << " microseconds" << std::endl;
    return result;
}

void GTEZ_DH(uint64_t xi, uint64_t d_xi, uint64_t deltai, uint64_t delta, int lx, std::vector<uint64_t>& ri, int party,
        std::vector<uint64_t>& r1i, std::vector<uint64_t>& bi, std::vector<uint64_t>& d_ri,
        std::vector<uint64_t>& d_r1i, std::vector<uint64_t>& d_bi,
        std::vector<std::vector<uint64_t>>& ci, int n)
{
    //auto start = std::chrono::high_resolution_clock::now();
    // {r0, ...rlx, r*, s, t}
    uint64_t* rands = new uint64_t[lx+4];
    k_P.random_data(rands, sizeof(uint64_t)*(lx+4));
    uint64_t* rands2 = new uint64_t[n-1];
    k_P.random_data(rands2, sizeof(uint64_t)*(n-1));
    // t
    uint64_t t = rands[lx+3] & 1;
    uint64_t d_ux = deltai;
    if (t == 1)
    {
        xi = xi * UINT64_MAX;
        d_xi = d_xi * UINT64_MAX;
        d_ux = d_ux * UINT64_MAX;
    }
    // u_*
    uint64_t ux = UINT64_MAX;
    for (int i = 0; i < n-1; i++)
    {
        ux -= rands2[i];
    }
    // v_*
    uint64_t vx = ux + 3 * xi;
    uint64_t d_vx = d_ux + 3 * d_xi - deltai;
    if (party == 1)
    {
        vx -= 1;
    }
    // array c
    uint64_t l2 = (1ULL << lx);
    uint64_t kl = (1ULL << (64-lx-1));
    uint64_t* arr_c = new uint64_t[lx];
    uint64_t* arr_d_c = new uint64_t[lx];
    for (int i = 0; i < lx; i++)
    {
        uint64_t m2 = (1ULL << i+1);
        arr_c[i] = kl * (xi + l2 * bi[i] + m2 * ri[i] + r1i[i]);
        arr_d_c[i] = kl * (d_xi + l2 * d_bi[i] + m2 * d_ri[i] + d_r1i[i]);
    }

    /* reconstruct c, simulate network cost */
    uint64_t comm = 64 * (n-1) * lx;
    COMM += comm * n;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }
    
    for (int i = 0; i < ci.size(); i++)
    {
        for (int j = 0; j < lx; j++)
        {
            arr_c[j] += ci[i][j];
        }
    }

    // compute truncation of  x
    uint64_t* c1 = new uint64_t[lx];
    uint64_t* cl = new uint64_t[lx];
    uint64_t* d_cl = new uint64_t[lx];
    uint64_t* v = new uint64_t[lx];
    uint64_t* d_v = new uint64_t[lx];
    uint64_t k = 64;
    for (int i = 0; i < lx; i++)
    {
        c1[i] = arr_c[i] >> (k - lx - 1);
        cl[i] = (arr_c[i] >> lx) & 1;
        d_cl[i] = deltai * cl[i];
        v[i] = bi[i] - 2 * cl[i] * bi[i];
        d_v[i] = d_bi[i] + d_cl[i] - 2 * cl[i] * d_bi[i];
        if (party == 1)
        {
            v[i] += cl[i];
        }
    }
    std::vector<uint64_t> wi;
    std::vector<uint64_t> d_wi;
    wi.resize(lx + 2);
    d_wi.resize(lx + 2);
    wi[0] = xi;
    d_wi[0] = d_xi;
    for (int i = 1; i <= lx; i++)
    {
        uint64_t m = i;
        uint64_t l_m = (1ULL << lx - m);
        c1[i-1] = c1[i-1] % (1ULL << lx);
        c1[i-1] = c1[i-1] >> m;
        wi[i] = -ri[i-1] + (1ULL << lx - m) * v[i-1];
        d_wi[i] = deltai * c1[i-1] - d_ri[i] + (1ULL << lx - m) * d_v[i];
        if (party == 1)
        {
            wi[i] += c1[i-1];
        }
    }
    
    for (int i = lx-1; i >= 0; i--)
    {
        wi[i] += wi[i+1];
        d_wi[i] += d_wi[i+1] - deltai;
        if (party == 1)
        {
            wi[i] -= 1;
        }
        wi[i] *= rands[i];
        d_wi[i] *= rands[i];
    }
    if (party == 1)
    {
        wi[lx] -= 1;
    }
    wi[lx] *= rands[lx];
    wi[lx+1] = vx * rands[lx+1];
    d_wi[lx] *= rands[lx];
    d_wi[lx+1] = d_vx * rands[lx+1];
    // shuffle
    std::mt19937 rng(rands[lx+2]);
    std::shuffle(wi.begin(), wi.end(), rng);
    std::shuffle(d_wi.begin(), d_wi.end(), rng);
    uint64_t c_dci = 0;
    for (int i = 0; i < lx; i++)
    {
        c_dci += deltai * arr_c[i] - arr_d_c[i];
    }

    /* Pn+1 reconstructs wi and verify {wi, d_wi, c_dc}, simulate network cost */
    comm = 64 * (lx + 2) * 2 + 64;
    COMM += comm * n;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }
    std::vector<std::vector<uint64_t>> w(n, std::vector<uint64_t>(lx+2, 0));
    std::vector<std::vector<uint64_t>> d_w(n, std::vector<uint64_t>(lx+2, 0));
    uint64_t ver1 = 0, ver2 = 0, c3 = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < lx+2; j++)
        {
            ver1 += w[i][j];
            ver2 += d_w[i][j];
        }
    }
    if (ver1 * delta != ver2)
    {
        // check failed
    }
    std::vector<uint64_t> cdc(n);
    for (int i = 0; i < n; i++)
    {
        c3 += cdc[i];
    }
    if (c3 != 0)
    {
        // check failed
    }
    uint64_t r = 0;
    uint64_t d_r = 0;
    for (int i = 0; i < lx+2; i++)
    {
        uint64_t temp = 0;
        for (int j = 0; j < n; j++)
        {
            temp += w[j][i];
        }
        if (temp == 0)
        {
            r = 1;
            break;
        }
    }
    uint64_t* random = new uint64_t[2];
    for (int i = 0; i < n-1; i++)
    {
        k_ij[i][n].random_data(random, sizeof(uint64_t) * 2);
        r -= random[0];
        d_r -= random[1];
    }

    // Pn+1 send [GTEZ(x)] to Pn, simulate network cost
    comm = 64 * 2;
    COMM += comm;
    
    if (LAN)
    {
        uint64_t lan_BW = 1024 * 1024 * 1024;
        uint64_t lan_delay = 500 + comm * 1000000 / lan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(lan_delay));
    }
    else
    {
        uint64_t wan_BW = 100 * 1024 * 1024;
        uint64_t wan_delay = 50000 + comm * 1000000 / wan_BW;
        std::this_thread::sleep_for(std::chrono::microseconds(wan_delay));
    }
    
    // Pi compute GTEZ(x)
    uint64_t rst = r - 2 * t * r;
    uint64_t d_rst = deltai * t + (1 - 2 * t) * r;
    if (party == 1)
    {
        rst += t;
    }
    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "DH Greater than or equal to zero execution time: " << duration << " microseconds" << std::endl;
    
    delete[] rands;
    delete[] rands2;
    delete[] arr_c;
    delete[] arr_d_c;
    delete[] c1;
    delete[] cl;
    delete[] d_cl;
    delete[] v;
    delete[] d_v;
    delete[] random;
}

int main()
{
    int n = 5;
    int party = 1;
    LAN = 1;
    std::vector<int> size = {565, 1115, 2215, 3865, 5515};
    std::vector<int> size2 = {58, 108, 208, 358, 508};
    std::vector<uint64_t> xi_ai(n-1);
    std::vector<uint64_t> yi_bi(n-1);
    uint64_t xi = 1;
    uint64_t yi = 1;
    uint64_t ai = 1;
    uint64_t bi = 1;
    uint64_t abi = 1;
    uint64_t lx = 32;
    std::vector<uint64_t> ri(lx);
    std::vector<uint64_t> r1i(lx);
    std::vector<uint64_t> arr_bi(lx);
    std::vector<std::vector<uint64_t>> ci(n-1, std::vector<uint64_t>(lx, 0));
    std::vector<std::vector<uint64_t>> w(n, std::vector<uint64_t>(lx+2, 0));
    initialize_PRG(n);
    MAC_SH(n);
    for (int i = 0; i < 5; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        BT_SH(n, size[i]);
        GAD_SH(n, size2[i]*60, 30, 15);
        for (int j = 0; j < size[i]; j++)
        {
            Mul_SH(xi, yi, ai, bi, abi, xi_ai, yi_bi, 0, n);
        }
        for (int j = 0; j < size2[i]; j++)
        {
            GTEZ_SH(xi, lx, ri, party, r1i, arr_bi, ci, w, n);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "SH execution time: " << duration << " microseconds" << std::endl;
        std::cout << "SH execution COMM: " << COMM / (8*1024) << " Kbits" << std::endl;
        COMM = 0;
    }
    
    COMM = 0;
    uint64_t vi, deltai, v, delta;
    uint64_t deltaxi, deltayi, delta_1, a1, b1, c1, ab1, ac1, bc1, abc1;
    uint64_t d_xi, d_ri, d_r1i, d_bi;
    std::vector<uint64_t> arr_d_ri(lx);
    std::vector<uint64_t> arr_d_r1i(lx);
    std::vector<uint64_t> arr_d_bi(lx);
    for (int i = 0; i < 5; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        BT_SH2(n, size[i]);
        GAD_SH2(delta, n, size2[i]*60, 30, 15);
        for (int j = 0; j < size[i]; j++)
        {
            Mul_DH(xi, yi, deltaxi, deltayi, delta_1, ai, bi, abi, a1, b1, c1, ab1, ac1, bc1, abc1, party, n);
        }
        for (int j = 0; j < size2[i]; j++)
        {
            GTEZ_DH(xi, d_xi, deltai, delta, lx, ri, party, r1i, arr_bi, arr_d_ri,arr_d_r1i, arr_d_bi, ci, n);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "execution time: " << duration << " microseconds" << std::endl;
        std::cout << "DH execution COMM: " << COMM << " bits" << std::endl;
        COMM = 0;
    }
    /*
    int n = 10;
    int party = 1;
    int size = 2*n+1;
    int size2 = 60*(2*n+1);
    int l = 30, m = 15;
    initialize_PRG(n);
    std::vector<uint64_t> inputs = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    SS_SH(inputs, n);
    BT_SH(n, size);

    std::vector<uint64_t> xi_ai(n-1);
    std::vector<uint64_t> yi_bi(n-1);
    uint64_t xi = 1;
    uint64_t yi = 1;
    uint64_t ai = 1;
    uint64_t bi = 1;
    uint64_t abi = 1;

    uint64_t xy;
    uint64_t s_r1i, s_ri;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 2*n+1; i++)
    {
        xy = Mul_SH(xi, yi, ai, bi, abi, xi_ai, yi_bi, 0, n);
        //Trunc_SH(xi, s_ri, s_r1i, bi, party, l, m, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "1000 fixed point Multiplication execution time: " << duration << " microseconds" << std::endl;

    GAD_SH(n, size2, 30, 15);
    //uint64_t s_r1i, s_ri;
    Trunc_SH(xi, s_ri, s_r1i, bi, party, l, m, n);

    uint64_t lx = 32;
    std::vector<uint64_t> ri(lx);
    std::vector<uint64_t> r1i(lx);
    std::vector<uint64_t> arr_bi(lx);
    std::vector<std::vector<uint64_t>> ci(n-1, std::vector<uint64_t>(lx, 0));
    std::vector<std::vector<uint64_t>> w(n, std::vector<uint64_t>(lx+2, 0));
    start = std::chrono::high_resolution_clock::now(); 
    for (int i = 0; i < 2*n+1; i++)
        GTEZ_SH(xi, lx, ri, party, r1i, arr_bi, ci, w, n);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "CMP execution time: " << duration << " microseconds" << std::endl;
    MAC_SH(n);
    challenge_Data_SH(1, n);

    uint64_t vi, deltai, v, delta;
    std::vector<uint64_t> arr_xi(1);
    std::vector<uint64_t> arr_d_xi(1);
    Verify_DH(arr_xi, arr_d_xi, vi, deltai, v, delta, n);

    std::vector<uint64_t> x(1);
    SS_SH2(x, deltai, delta, party, n);
    BT_SH2(n, size);
    
    uint64_t deltaxi, deltayi, delta_1, a1, b1, c1, ab1, ac1, bc1, abc1;
    uint64_t d_xi, d_ri, d_r1i, d_bi;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 2*n+1; i++)
    {
        Mul_DH(xi, yi, deltaxi, deltayi, delta_1, ai, bi, abi, a1, b1, c1, ab1, ac1, bc1, abc1, party, n);
        //Trunc_DH(xi, s_ri, s_r1i, bi, d_xi, d_ri, d_r1i, d_bi, deltai, party, l, m, n);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "1000 DH fixed point Multiplication execution time: " << duration << " microseconds" << std::endl;
    
    GAD_SH2(delta, n, size2, l, m);
    
    //uint64_t d_xi, d_ri, d_r1i, d_bi;
    Trunc_DH(xi, s_ri, s_r1i, bi, d_xi, d_ri, d_r1i, d_bi, deltai, party, l, m, n);

    std::vector<uint64_t> arr_d_ri(lx);
    std::vector<uint64_t> arr_d_r1i(lx);
    std::vector<uint64_t> arr_d_bi(lx);
    start = std::chrono::high_resolution_clock::now(); 
    for (int i = 0; i < 2*n+1; i++)
        GTEZ_DH(xi, d_xi, deltai, delta, lx, ri, party, r1i, arr_bi, arr_d_ri,arr_d_r1i, arr_d_bi, ci, n);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "CMP execution time: " << duration << " microseconds" << std::endl;
    */
    // std::string a[10];
    // a[0] = "1111111111111111";
    // PRG prg(static_cast<const void*>(a[0].c_str()));//using a secure random seed

    // int rand_int, rand_ints[100];
    // rand_int = 0;
    // block rand_block[3];

    // prg.random_data(&rand_int, sizeof(rand_int)); //fill rand_int with 32 random bits
    // prg.random_block(rand_block, 3);	      //fill rand_block with 128*3 random bits

    // prg.reseed(&rand_block[1]);                   //reset the seed and counter in prg
    // prg.random_data_unaligned(rand_ints+2, sizeof(int)*98);  //when the array is not 128-bit-aligned
    
    // for (int i = 0 ;i < 10; i++)
    //{
        //std::cout<< rand_ints[i] << std::endl;
    //}
    //uint64_t aa = 0;
    //std::cout << aa-1 << std::endl;
    // printMatrix(100);
    return 0;
}
