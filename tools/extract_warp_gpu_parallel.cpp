#include "dense_flow.h"
#include "utils.h"
#include "path_tools.h"

INITIALIZE_EASYLOGGINGPP

using namespace cv::cuda;
namespace tt = std::chrono;

int excuteJob(std::string vidFile, std::string dump_base, int bound, int type, int step, int device_id)
{
    std::string relative_path = pathSplitWithIdx(vidFile, -4, 0);
    std::string dump_path = safelyJoinPath(dump_base, relative_path);
    std::string lock_path = dump_path + ".lock";

    if (checkDirs(lock_path) == 0 || makeDirs(dump_path) != 0){
        //lock
        std::ofstream f_lock;
        f_lock.open(lock_path);

        // std::cout << dump_path << std::endl;
        try{
            calcDenseWarpFlowGPU(vidFile, dump_path, bound, type, step, device_id);
        }
        catch (cv::Exception){
            return -1;
        }
        // unlock
        f_lock.close();
        remove((lock_path).c_str());

        return 0;
    }

    return -1;
}

int main(int argc, char** argv){
	// IO operation
	const char* keys =
		{
			"{ f dsfile_path | ds.txt  | filelist of video }"
            "{ d dump_base   | data    | dump base path of flow }"
            "{ n num_process | 8       | number of process}"
			"{ b bound       | 20      | specify the maximum of optical flow}"
			"{ t type        | 0       | specify the optical flow algorithm }"
			"{ s step        | 1       | specify the step for frame sampling}"
            "{ g num_gpu     | 8       | number of gpu}"
		};

	CommandLineParser cmd(argc, argv, keys);
	std::string dsfile_path = cmd.get<std::string>("dsfile_path");
    std::string dump_base = cmd.get<std::string>("dump_base");
    int num_process = cmd.get<int>("num_process");
	int bound = cmd.get<int>("bound");
    int type  = cmd.get<int>("type");
    int step = cmd.get<int>("step");
    int num_gpu = cmd.get<int>("num_gpu");

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    std::cout << "num process to extract flow: " << num_process << std::endl;

    vector<string> job_vec;
    auto lines = parseDS(dsfile_path, job_vec);
    std::cout << "num videos: " << job_vec.size() << std::endl;
    std::cout << "dump base:  " << dump_base << std::endl;

    int rank_id = 0;
    pid_t c_pid;
    for (int i = 1; i < num_process; i++){
        c_pid = fork();
        if (c_pid == 0){
            // important exit
            rank_id = i;
            break;
        }
    }

    /* * * * * * * * * * * * * * * *
     * below is child process code *
     * * * * * * * * * * * * * * * */
    int start_idx = rank_id * lines / num_process;
    int end_idx = rank_id == (num_process - 1) ? lines : (rank_id + 1) * lines / num_process;
    int gpu_id = rank_id % num_gpu;

    printf("process %d ready! getting jobs %d-%d, total %d, using gpu %d\n", rank_id, start_idx, end_idx, end_idx - start_idx, gpu_id);
    fflush(stdout);
    usleep(10000);

    int t_accu = 0;
    int t_avg = 0;
    int n_pres = 0;

    for (int i = start_idx; i < end_idx; i++){
        tt::steady_clock::time_point begin = tt::steady_clock::now();

        string job = job_vec[i];
        excuteJob(job, dump_base, bound, type, step, gpu_id);

        tt::steady_clock::time_point end = tt::steady_clock::now();
        auto t_iter = tt::duration_cast<tt::milliseconds>(end - begin).count();

        if (t_iter < 500){
            n_pres++;
        }

        if (rank_id == 0){
            //usleep(1000000);
            int n_done = i - start_idx + 1;
            int n_total = end_idx - start_idx;

            t_accu += t_iter;
            t_avg = t_accu / (1e-5 + n_done - n_pres);

            int t_eta = (n_total - n_done) * t_avg;
            printf("jobs num: [done/total]: %d/%d | time: [eta/avg/pass]: %.1fs/%.1fs/%.1fs       \r", n_done, n_total, t_eta/1000.0, t_avg/1000.0, t_accu/1000.0);
            fflush(stdout);
        }
    }

	return 0;
}
