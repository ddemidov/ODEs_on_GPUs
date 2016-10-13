#ifndef LOG_HPP
#define LOG_HPP

#include <fstream>
#include <boost/timer/timer.hpp>

void log_perf(const char *version, size_t size, size_t iter, boost::timer::cpu_times time) {
    std::ofstream f("perf.dat", std::ios::app);
    f << version << "\t" << size << "\t" << iter << "\t" << boost::timer::format(time, 8, "%w") << "\n";
}

#endif
