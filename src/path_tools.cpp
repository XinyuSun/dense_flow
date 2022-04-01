#include "path_tools.h"

void split(const std::string& s, std::vector<std::string>& tokens, const std::string& delimeters = " ")
{
    std::string::size_type lastPos = s.find_first_not_of(delimeters, 0);
    std::string::size_type pos = s.find_first_of(delimeters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
        tokens.push_back(s.substr(lastPos, pos - lastPos));
        lastPos = s.find_first_not_of(delimeters, pos);
        pos = s.find_first_of(delimeters, lastPos);
    }
}

void join(const std::vector<std::string>& v, const std::string c, std::string& s)
{
   s.clear();

   for (std::vector<std::string>::const_iterator p = v.begin();
        p != v.end(); ++p) {
      s += *p;
      if (p != v.end() - 1)
        s += c;
   }
}

std::string pathSplitWithIdx(const std::string c, int start, int end)
{
    std::vector<std::string> tokens;
    std::vector<std::string> sub_tokens;

    split(c, tokens, "/");

    auto it_start = start < 0 ? tokens.end() + start : tokens.begin() + start;
    auto it_end = end <= 0 ? tokens.end() + end : tokens.begin() + end;
    
    sub_tokens.assign(it_start, it_end);

    std::string cc;
    join(sub_tokens, "/", cc);

    if (it_start == tokens.begin() && c[0] == '/'){
        cc = "/" + cc;
    }
    
    return cc;
}

std::string safelyJoinPath(std::string c1, std::string c2)
{
    if (c1.at(c1.length() - 1) != '/'){
        c1 += "/";
    }
    return c1 + c2;
}

int checkDirs(std::string prefix)
{
    // 0 is exist
    return access(prefix.c_str(), F_OK);
}

int makeDirs(std::string prefix)
{
    int ret;
    // cout << prefix << endl;
    if (checkDirs(prefix) != 0){
        // folder is not exist
        ret = mkdir(prefix.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (ret != 0){
            return makeDirs(pathSplitWithIdx(prefix, 0, -1));
        }
        else{
            // create folder successfully
            return -1;
        }
    }else{
        // folder exist
        return 0;
    }
}

int parseDS(std::string dsfile_path, std::vector<std::string>& job_vec)
{
    std::ifstream fin;
    fin.open(dsfile_path);
    int num_lines = 0;
    while(!fin.eof()){
        std::string line;
        getline(fin, line);
        if (line.length() > 1){
            job_vec.push_back(line);
            num_lines++;
        }
    }
    return num_lines;
}
