#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <chrono>
#include <vector>
#include <unistd.h>
#include <time.h>
#include <cstdio>
#include <sys/stat.h>

void split(const std::string& s, std::vector<std::string>& tokens, const std::string& delimeters);

void join(const std::vector<std::string>& v, const std::string c, std::string& s);

std::string pathSplitWithIdx(const std::string c, int start, int end);

std::string safelyJoinPath(std::string c1, std::string c2);

int checkDirs(std::string prefix);

int makeDirs(std::string prefix);

int parseDS(std::string dsfile_path, std::vector<std::string>& job_vec);