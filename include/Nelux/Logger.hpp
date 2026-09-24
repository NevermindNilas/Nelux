// src/logger.h

#ifndef CELEX_LOGGER_H
#define CELEX_LOGGER_H

#include <memory>
#include <spdlog/common.h>

// Compile-time log level gate. Code below this level is compiled out.
// Default warn; Release (NDEBUG) defaults to off. Override with
// -DNELUX_COMPILED_LOG_LEVEL=SPDLOG_LEVEL_DEBUG etc.
#ifndef NELUX_COMPILED_LOG_LEVEL
#ifdef NDEBUG
#define NELUX_COMPILED_LOG_LEVEL SPDLOG_LEVEL_OFF
#else
#define NELUX_COMPILED_LOG_LEVEL SPDLOG_LEVEL_WARN
#endif
#endif

// Tell spdlog headers the same thing so its own macros agree with ours.
#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL NELUX_COMPILED_LOG_LEVEL
#endif

// Lightweight header: forward-declare spdlog::logger instead of pulling in
// <spdlog/spdlog.h> (header-only spdlog is intentional — no compiled-spdlog
// switch, no unity build). <spdlog/common.h> supplies level_enum cheaply;
// the shared_ptr to an incomplete logger is valid, and Logger.cpp plus any
// TU that expands the NELUX_* macros below includes <spdlog/spdlog.h>
// explicitly for the complete type.

namespace spdlog
{
class logger;
}

namespace nelux
{

class Logger
{
  public:
    // Retrieves the singleton instance
    static std::shared_ptr<spdlog::logger>& get_logger();
    // Cached raw pointer for hot macros (no shared_ptr refcount churn).
    static spdlog::logger* get_raw();

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


//conveniece macros (do{...}while(0) so they behave as one statement).
// NOTE: expanding these requires the complete spdlog::logger type, so any
// TU using them must include <spdlog/spdlog.h> (Logger.cpp does; all other
// logging TUs do too). The fwd-decl above keeps non-logging includers light.
// Hot paths call get_raw() (cached pointer, no shared_ptr churn).
#define NELUX_TRACE(...)                                                       \
    do                                                                         \
    {                                                                          \
        auto* _nl = ::nelux::Logger::get_raw();                                \
        if (_nl)                                                               \
            _nl->trace(__VA_ARGS__);                                           \
    } while (0)
#define NELUX_DEBUG(...)                                                       \
    do                                                                         \
    {                                                                          \
        auto* _nl = ::nelux::Logger::get_raw();                                \
        if (_nl)                                                               \
            _nl->debug(__VA_ARGS__);                                           \
    } while (0)
#define NELUX_INFO(...)                                                        \
    do                                                                         \
    {                                                                          \
        auto* _nl = ::nelux::Logger::get_raw();                                \
        if (_nl)                                                               \
            _nl->info(__VA_ARGS__);                                            \
    } while (0)
#define NELUX_WARN(...)                                                        \
    do                                                                         \
    {                                                                          \
        auto* _nl = ::nelux::Logger::get_raw();                                \
        if (_nl)                                                               \
            _nl->warn(__VA_ARGS__);                                            \
    } while (0)
#define NELUX_ERROR(...)                                                       \
    do                                                                         \
    {                                                                          \
        auto* _nl = ::nelux::Logger::get_raw();                                \
        if (_nl)                                                               \
            _nl->error(__VA_ARGS__);                                           \
    } while (0)
#define NELUX_CRITICAL(...)                                                    \
    do                                                                         \
    {                                                                          \
        auto* _nl = ::nelux::Logger::get_raw();                                \
        if (_nl)                                                               \
            _nl->critical(__VA_ARGS__);                                        \
    } while (0)



#endif // CELEX_LOGGER_H
