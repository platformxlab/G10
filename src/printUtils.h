#ifndef __PRINT_UTILS_H__
#define __PRINT_UTILS_H__

// debug
#define dprintf(format, ...) \
    printf(("\033[96m" + std::string(format) + "\033[0m").c_str(), __VA_ARGS__)
// info
#define iprintf(format, ...) \
    printf(("\033[92m" + std::string(format) + "\033[0m").c_str(), __VA_ARGS__)
// notice
#define nprintf(format, ...) \
    printf(("\033[94m" + std::string(format) + "\033[0m").c_str(), __VA_ARGS__)
// warning
#define wprintf(format, ...) \
    printf(("\033[93m" + std::string(format) + "\033[0m").c_str(), __VA_ARGS__)
// error
#define eprintf(format, ...) \
    printf(("\033[91m" + std::string(format) + "\033[0m").c_str(), __VA_ARGS__)

#endif
