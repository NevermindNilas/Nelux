// src/logger.cpp


#include "Logger.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>

namespace nelux
{

// Initialize the static logger instance
std::shared_ptr<spdlog::logger> Logger::logger_instance = nullptr;

std::shared_ptr<spdlog::logger>& Logger::get_logger()
{
    if (!logger_instance)
    {
        // Create a console logger with color
        logger_instance = spdlog::stdout_color_mt("nelux");
        logger_instance->set_level(spdlog::level::off); // Default level
        logger_instance->set_pattern("[%^%Y-%m-%d %H:%M:%S%$] [%l] %v");
    }
    return logger_instance;
}

void Logger::set_level(spdlog::level::level_enum level)
{
    get_logger()->set_level(level);
}

} // namespace nelux
