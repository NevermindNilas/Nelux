// src/logger.h

#ifndef CELEX_LOGGER_H
#define CELEX_LOGGER_H

#include <memory>
#include <spdlog/spdlog.h>

namespace nelux
{

class Logger
{
  public:
    // Retrieves the singleton instance
    static std::shared_ptr<spdlog::logger>& get_logger();

    // Configures the logger's verbosity
    static void set_level(spdlog::level::level_enum level);

  private:
    Logger() = default;
    ~Logger() = default;

    // Deleted to prevent copying
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    static std::shared_ptr<spdlog::logger> logger_instance;
};

} // namespace nelux


//conveniece macros
#define NELUX_TRACE(...)                                                       \
    if (nelux::Logger::get_logger()->should_log(spdlog::level::trace))         \
    nelux::Logger::get_logger()->trace(__VA_ARGS__)
#define NELUX_DEBUG(...)                                                       \
    if (nelux::Logger::get_logger()->should_log(spdlog::level::debug))         \
    nelux::Logger::get_logger()->debug(__VA_ARGS__)
#define NELUX_INFO(...)                                                        \
    if (nelux::Logger::get_logger()->should_log(spdlog::level::info))          \
    nelux::Logger::get_logger()->info(__VA_ARGS__)
#define NELUX_WARN(...)                                                        \
    if (nelux::Logger::get_logger()->should_log(spdlog::level::warn))          \
    nelux::Logger::get_logger()->warn(__VA_ARGS__)
#define NELUX_ERROR(...)                                                       \
    if (nelux::Logger::get_logger()->should_log(spdlog::level::err))           \
    nelux::Logger::get_logger()->error(__VA_ARGS__)
#define NELUX_CRITICAL(...)                                                    \
    if (nelux::Logger::get_logger()->should_log(spdlog::level::critical))      \
    nelux::Logger::get_logger()->critical(__VA_ARGS__)



#endif // CELEX_LOGGER_H
