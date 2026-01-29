// Filter.hpp

#ifndef FILTER_HPP
#define FILTER_HPP

#include <libavfilter/avfilter.h>
#include <string>

class Filter
{
  public:
    /**
     * @brief Constructs a Filter with a given name and options.
     *
     * @param name Name of the filter (e.g., "scale", "hue").
     * @param options Options for the filter in key=value format or positional
     * arguments.
     */
    Filter(const std::string& name, const std::string& options = "")
        : name_(name), options_(options), is_valid_(false)
    {
        is_valid_ = isFilterAvailable(name_);
    }

    /**
     * @brief Retrieves the name of the filter.
     *
     * @return const std::string& Filter name.
     */
    const std::string& getName() const
    {
        return name_;
    }

    /**
     * @brief Retrieves the options of the filter.
     *
     * @return const std::string& Filter options.
     */
    const std::string& getOptions() const
    {
        return options_;
    }

    /**
     * @brief Generates the filter description string compatible with FFmpeg.
     *        If options are present, returns "name=options", else returns "name".
     *
     * @return std::string FFmpeg filter description.
     */
    std::string getFilterDescription() const
    {
        if (options_.empty())
        {
            return name_;
        }
        else
        {
            return name_ + "=" + options_;
        }
    }

    /**
     * @brief Checks if the filter is available in the current FFmpeg build.
     *
     * @return true If the filter exists.
     * @return false If the filter does not exist.
     */
    bool isValid() const
    {
        return is_valid_;
    }

    /**
     * @brief Static method to check if a filter with the given name exists.
     *
     * @param name Name of the filter to check.
     * @return true If the filter exists.
     * @return false If the filter does not exist.
     */
    static bool isFilterAvailable(const std::string& name)
    {
        return avfilter_get_by_name(name.c_str()) != nullptr;
    }

  private:
    std::string name_;    // Filter name
    std::string options_; // Filter options
    bool is_valid_;       // Validity flag
};

#endif // FILTER_HPP
