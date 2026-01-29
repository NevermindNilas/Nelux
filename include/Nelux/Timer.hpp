#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>
#include <string>

class Timer
{
  public:
    Timer() : m_start(std::chrono::high_resolution_clock::now())
    {
    }

    void reset()
    {
        m_start = std::chrono::high_resolution_clock::now();
    }

    void print(const std::string& message)
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - m_start;
        std::cout << message << " " << elapsed.count() << "s" << std::endl;
    }

    double elapsed()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - m_start;
        m_end = end;
        return elapsed.count();
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start, m_end;
};

#endif
